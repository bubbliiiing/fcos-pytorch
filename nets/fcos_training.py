import math
from functools import partial

import torch
import torch.nn as nn


class Fcos_Loss(nn.Module):
    def __init__(self, strides = [8, 16, 32, 64, 128], limit_range = [[-1,64],[64,128],[128,256],[256,512],[512,999999]], sample_radiu_ratio=1.5):
        super().__init__()
        self.strides            = strides
        self.limit_range        = limit_range
        self.sample_radiu_ratio = sample_radiu_ratio
        assert len(strides)==len(limit_range)
    
    def forward(self, preds, gt_boxes, classes):
        cls_logits, cnt_logits, reg_preds       = preds
        #---------------------------------#
        #   获得应该有的预测目标
        #---------------------------------#
        cls_targets, cnt_targets, reg_targets   = self.get_target(cls_logits, gt_boxes=gt_boxes, classes=classes)
        
        #---------------------------------#
        #   找到正样本
        #---------------------------------#
        mask_pos = (cnt_targets > -1).squeeze(dim=-1)
        cls_loss = self.compute_cls_loss(cls_logits, cls_targets, mask_pos)
        cnt_loss = self.compute_cnt_loss(cnt_logits, cnt_targets, mask_pos)
        reg_loss = self.compute_reg_loss(reg_preds, reg_targets, mask_pos)

        total_loss = cls_loss + cnt_loss + reg_loss
        return total_loss

    def _get_grids(self, pred, stride):
        h, w     = pred.shape[2:4]
        shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
        shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

        shift_x = torch.reshape(shift_x, [-1])
        shift_y = torch.reshape(shift_y, [-1])
        grid    = torch.stack([shift_x, shift_y], -1) + stride // 2

        return grid

    def get_target(self, cls_logits, gt_boxes, classes):
        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []

        for level in range(len(cls_logits)):
            cls_logit   = cls_logits[level]
            stride      = self.strides[level]
            limit_range = self.limit_range[level]
            #--------------------#
            #   计算batch_size
            #   计算种类数量
            #--------------------#
            batch_size  = cls_logit.shape[0]
            num_classes = cls_logit.shape[1]

            #-----------------------#
            #   获得网格
            #-----------------------#
            grids       = self._get_grids(cls_logit, stride)
            grids       = grids.type_as(cls_logit)
            x           = grids[:, 0]
            y           = grids[:, 1]
                
            #-------------------------------------#
            #   对预测结果进行reshape
            #   [batch_size, h * w, num_classes] 
            #-------------------------------------#
            cls_logit   = cls_logit.permute(0, 2, 3, 1).reshape((batch_size, -1, num_classes))
            h_mul_w     = cls_logit.shape[1]

            #----------------------------------------------------------------#
            #   左上点、右下点不可以差距很大
            #   求真实框的左上角和右下角相比于特征点的偏移情况
            #   [1, h*w, 1] - [batch_size, 1, m] --> [batch_size, h*w, m]
            #----------------------------------------------------------------#
            left_off    = x[None, :, None] - gt_boxes[...,0][:, None, :]
            top_off     = y[None, :, None] - gt_boxes[...,1][:, None, :]
            right_off   = gt_boxes[..., 2][:, None, :] - x[None, :, None]
            bottom_off  = gt_boxes[..., 3][:, None, :] - y[None, :, None]
            #----------------------------------------------------------------#
            #   [batch_size, h*w, m, 4]
            #----------------------------------------------------------------#
            ltrb_off    = torch.stack([left_off, top_off, right_off, bottom_off],dim=-1)
            
            #----------------------------------------------------------------#
            #   求每个框的面积
            #   [batch_size, h*w, m]
            #----------------------------------------------------------------#
            areas       = (ltrb_off[...,0] + ltrb_off[...,2]) * (ltrb_off[...,1] + ltrb_off[...,3])
            #----------------------------------------------------------------#
            #   [batch_size,h*w,m]
            #----------------------------------------------------------------#
            off_min     = torch.min(ltrb_off, dim=-1)[0]
            off_max     = torch.max(ltrb_off, dim=-1)[0]

            #----------------------------------------------------------------#
            #   将特征点不落在真实框内的特征点剔除。
            #   浅层特征适合小目标检测，深层特征适合大目标检测。
            #----------------------------------------------------------------#
            mask_in_gtboxes = off_min > 0
            mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])

            radiu       = stride * self.sample_radiu_ratio
            #----------------------------------------------------------------#
            #   中心点不可以差距很大，求真实框中心相比于特征点的偏移情况
            #   计算真实框中心的x轴坐标
            #   计算真实框中心的y轴坐标
            #   [1,h*w,1] - [batch_size, 1, m] --> [batch_size,h * w, m]
            #----------------------------------------------------------------#
            gt_center_x = (gt_boxes[...,0] + gt_boxes[...,2]) / 2
            gt_center_y = (gt_boxes[...,1] + gt_boxes[...,3]) / 2
            c_left_off      = x[None, :, None] - gt_center_x[:, None, :]
            c_top_off       = y[None, :, None] - gt_center_y[:, None, :]
            c_right_off     = gt_center_x[:, None, :] - x[None, :, None]
            c_bottom_off    = gt_center_y[:, None, :] - y[None, :, None]
            #----------------------------------------------------------------#
            #   [batch_size, h*w, m, 4]
            #----------------------------------------------------------------#
            c_ltrb_off  = torch.stack([c_left_off, c_top_off, c_right_off, c_bottom_off],dim=-1)
            c_off_max   = torch.max(c_ltrb_off,dim=-1)[0]
            mask_center = c_off_max < radiu

            #----------------------------------------------------------------#
            #   为正样本的特征点
            #   [batch_size, h*w, m]
            #----------------------------------------------------------------#
            mask_pos    = mask_in_gtboxes & mask_in_level & mask_center

            #----------------------------------------------------------------#
            #   将所有不是正样本的特征点，面积设成max
            #   [batch_size, h*w, m]
            #----------------------------------------------------------------#
            areas[~mask_pos]    = 99999999
            #----------------------------------------------------------------#
            #   选取该特征点对应面积最小的框
            #   [batch_size, h*w]
            #----------------------------------------------------------------#
            areas_min_ind       = torch.min(areas, dim = -1)[1]
            #----------------------------------------------------------------#
            #   [batch_size*h*w, 4]
            #----------------------------------------------------------------#
            reg_targets = ltrb_off[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
            reg_targets = torch.reshape(reg_targets, (batch_size, -1, 4))
            #----------------------------------------------------------------#
            #   [batch_size, h*w, m]
            #----------------------------------------------------------------#
            _classes    = torch.broadcast_tensors(classes[:, None, :], areas.long())[0]
            cls_targets = _classes[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
            #----------------------------------------------------------------#
            #   [batch_size, h*w, 1]
            #----------------------------------------------------------------#
            cls_targets = torch.reshape(cls_targets,(batch_size,-1,1))

            #----------------------------------------------------------------#
            #   [batch_size, h*w]
            #----------------------------------------------------------------#
            left_right_min  = torch.min(reg_targets[..., 0], reg_targets[..., 2])
            left_right_max  = torch.max(reg_targets[..., 0], reg_targets[..., 2])
            top_bottom_min  = torch.min(reg_targets[..., 1], reg_targets[..., 3])
            top_bottom_max  = torch.max(reg_targets[..., 1], reg_targets[..., 3])
            #----------------------------------------------------------------#
            #   [batch_size, h*w, 1]
            #----------------------------------------------------------------#
            cnt_targets= ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(dim=-1)

            assert reg_targets.shape == (batch_size,h_mul_w,4)
            assert cls_targets.shape == (batch_size,h_mul_w,1)
            assert cnt_targets.shape == (batch_size,h_mul_w,1)

            #----------------------------------------------------------------#
            #   process neg grids
            #----------------------------------------------------------------#
            mask_pos_2 = mask_pos.long().sum(dim=-1) >= 1
            assert mask_pos_2.shape  == (batch_size,h_mul_w)
            cls_targets[~mask_pos_2] = -1
            cnt_targets[~mask_pos_2] = -1
            reg_targets[~mask_pos_2] = -1

            cls_targets_all_level.append(cls_targets)
            cnt_targets_all_level.append(cnt_targets)
            reg_targets_all_level.append(reg_targets)
            
        return torch.cat(cls_targets_all_level, dim=1), torch.cat(cnt_targets_all_level, dim=1), torch.cat(reg_targets_all_level, dim=1)
    
    def compute_cls_loss(self, preds, targets, mask, gamma=2.0, alpha=0.25):
        #--------------------#
        #   计算batch_size
        #   计算种类数量
        #--------------------#
        batch_size      = targets.shape[0]
        num_classes     = preds[0].shape[1]
        
        mask            = mask.unsqueeze(dim = -1)
        #--------------------#
        #   计算正样本数量
        #--------------------#
        num_pos         = torch.sum(mask, dim = [1, 2]).clamp_(min = 1).float()
        preds_reshape   = []
        for pred in preds:
            #--------------------#
            #   对预测结果reshape
            #--------------------#
            pred        = torch.reshape(pred.permute(0, 2, 3, 1), [batch_size, -1, num_classes])
            preds_reshape.append(pred)
        preds           = torch.cat(preds_reshape, dim = 1)
        assert preds.shape[:2]==targets.shape[:2]
        
        #--------------------#
        #   对计算损失
        #--------------------#
        loss = 0
        for batch_index in range(batch_size):
            pred_pos    = torch.sigmoid(preds[batch_index])
            target_pos  = targets[batch_index]
            #--------------------#
            #   生成one_hot标签
            #--------------------#
            target_pos  = (torch.arange(0, num_classes, device=target_pos.device)[None,:] == target_pos).float()
            
            #--------------------#
            #   计算focal_loss
            #--------------------#
            pt          = pred_pos * target_pos + (1.0 - pred_pos) * (1.0 - target_pos)
            w           = alpha * target_pos + (1.0 - alpha) * (1.0 - target_pos)
            batch_loss  = -w * torch.pow((1.0 - pt), gamma) * pt.log()
            batch_loss  = batch_loss.sum()
            loss += batch_loss
            
        return loss / torch.sum(num_pos)

    def compute_cnt_loss(self, preds, targets, mask):
        #------------------------#
        #   计算batch_size
        #   计算center长度（1）
        #------------------------#
        batch_size  = targets.shape[0]
        c           = targets.shape[-1]
        
        mask            = mask.unsqueeze(dim = -1)
        #--------------------#
        #   计算正样本数量
        #--------------------#
        num_pos         = torch.sum(mask, dim = [1, 2]).clamp_(min = 1).float()
        
        preds_reshape   = []
        for pred in preds:
            #--------------------#
            #   对预测结果reshape
            #--------------------#
            pred        = torch.reshape(pred.permute(0, 2, 3, 1), [batch_size, -1, c])
            preds_reshape.append(pred)
            
        preds           = torch.cat(preds_reshape, dim = 1)
        assert preds.shape==targets.shape
        
        #--------------------#
        #   对计算损失
        #--------------------#
        loss = 0
        for batch_index in range(batch_size):
            pred_pos    = preds[batch_index][mask[batch_index]]
            target_pos  = targets[batch_index][mask[batch_index]]
            batch_loss  = nn.functional.binary_cross_entropy_with_logits(input=pred_pos,target=target_pos,reduction='sum').view(1)
            loss += batch_loss
            
        return torch.sum(loss, dim=0) / torch.sum(num_pos)

    def giou_loss(self, preds, targets):
        #------------------------#
        #   左上角和右下角
        #------------------------#
        lt_min  = torch.min(preds[:, :2], targets[:, :2])
        rb_min  = torch.min(preds[:, 2:], targets[:, 2:])
        #------------------------#
        #   重合面积计算
        #------------------------#
        wh_min  = (rb_min + lt_min).clamp(min=0)
        overlap = wh_min[:, 0] * wh_min[:, 1]#[n]
        
        #------------------------------#
        #   预测框面积和实际框面积计算
        #------------------------------#
        area1   = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
        area2   = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
        
        #------------------------------#
        #   计算交并比
        #------------------------------#
        union   = (area1 + area2 - overlap)
        iou     = overlap / union

        #------------------------------#
        #   计算外包围框
        #------------------------------#
        lt_max  = torch.max(preds[:, :2],targets[:, :2])
        rb_max  = torch.max(preds[:, 2:],targets[:, 2:])
        wh_max  = (rb_max + lt_max).clamp(0)
        G_area  = wh_max[:, 0] * wh_max[:, 1]

        #------------------------------#
        #   计算GIOU
        #------------------------------#
        giou    = iou - (G_area - union) / G_area.clamp(1e-10)
        loss    = 1. - giou
        return loss.sum()
        
    def compute_reg_loss(self, preds, targets, mask):
        #------------------------#
        #   计算batch_size
        #   计算回归参数长度（4）
        #------------------------#
        batch_size  = targets.shape[0]
        c           = targets.shape[-1]
        
        num_pos     = torch.sum(mask, dim=1).clamp_(min=1).float()#[batch_size,]
        preds_reshape=[]
        for pred in preds:
            #--------------------#
            #   对预测结果reshape
            #--------------------#
            pred        = torch.reshape(pred.permute(0, 2, 3, 1), [batch_size, -1, c])
            preds_reshape.append(pred)
            
        preds           = torch.cat(preds_reshape, dim = 1)
        assert preds.shape==targets.shape
        
        loss = 0
        for batch_index in range(batch_size):
            pred_pos    = preds[batch_index][mask[batch_index]]
            target_pos  = targets[batch_index][mask[batch_index]]
            batch_loss  = self.giou_loss(pred_pos, target_pos).view(1)
            loss += batch_loss
        return torch.sum(loss, dim=0) / torch.sum(num_pos)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
