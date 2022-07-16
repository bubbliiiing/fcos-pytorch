import numpy as np
import torch
from torchvision.ops import nms


class DecodeBox():
    def __init__(self, strides=[8, 16, 32, 64, 128]):
        super().__init__()
        self.strides=strides

    def _reshape_cat_out(self, inputs):
        out=[]
        for pred in inputs:
            pred    = pred.permute(0, 2, 3, 1)
            pred    = torch.reshape(pred, [inputs[0].shape[0], -1, inputs[0].shape[1]])
            out.append(pred)
        return torch.cat(out, dim=1)
    
    def decode_box(self, inputs, input_shape):
        #--------------------------------------#
        #   获得分类参数
        #--------------------------------------#
        cls_logits  = inputs[0]
        #--------------------------------------#
        #   获得中心偏移参数
        #--------------------------------------#
        cnt_logits  = inputs[1]
        #--------------------------------------#
        #   获得回归参数
        #--------------------------------------#
        reg_preds   = inputs[2]
        
        cls_preds   = self._reshape_cat_out(cls_logits)
        cnt_preds   = self._reshape_cat_out(cnt_logits)
        reg_preds   = self._reshape_cat_out(reg_preds)

        cls_preds  = torch.sigmoid(cls_preds)
        cnt_preds  = torch.sigmoid(cnt_preds)
        
        #--------------------------------------#
        #   对网格进行循环
        #--------------------------------------#
        grids = []
        for pred, stride in zip(inputs[0], self.strides):
            h, w     = pred.shape[2:4]
            shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
            shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

            shift_x = torch.reshape(shift_x, [-1])
            shift_y = torch.reshape(shift_y, [-1])
            grid    = torch.stack([shift_x, shift_y], -1) + stride // 2

            grids.append(grid)
        grids = torch.cat(grids, dim=0)
        if cls_preds.is_cuda:
            grids = grids.cuda()

        #--------------------------------------#
        #   获得种类与得分，与中心偏移参数相乘
        #--------------------------------------#
        cls_scores, cls_classes = torch.max(cls_preds,dim=-1)
        cls_scores              = torch.sqrt(cls_scores*(cnt_preds.squeeze(dim=-1)))

        #--------------------------------------#
        #   获得左上角右下角
        #--------------------------------------#
        left_top        = grids[None, :, :] - reg_preds[..., :2]
        right_bottom    = grids[None, :, :] + reg_preds[..., 2:]
        
        boxes               = torch.cat([left_top, right_bottom], dim=-1)
        boxes[..., [0, 2]]  = boxes[..., [0, 2]] / input_shape[1]
        boxes[..., [1, 3]]  = boxes[..., [1, 3]] / input_shape[0]
        boxes_score_classes = torch.cat([boxes, torch.unsqueeze(cls_scores.float(),-1), torch.unsqueeze(cls_classes.float(),-1)], dim=-1)
        return boxes_score_classes

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes
    
    def non_max_suppression(self, prediction, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            #----------------------------------------------------------#
            #   获得置信度
            #   利用置信度进行第一轮筛选
            #----------------------------------------------------------#
            class_conf = image_pred[:, 4:5]
            conf_mask  = (class_conf[:, 0] >= conf_thres).squeeze()

            #----------------------------------------------------------#
            #   detections的内容为
            #   x1, y1, x2, y2, class_conf, class_pred
            #----------------------------------------------------------#
            detections = image_pred[conf_mask]
            if not image_pred.size(0):
                continue

            #----------------------------------------------------------#
            #   获得种类
            #----------------------------------------------------------#
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                # 获得某一类初步筛选后全部的预测结果
                detections_class = detections[detections[:, -1] == c]

                #------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #------------------------------------------#
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4],
                    nms_thres
                )
                max_detections = detections_class[keep]
                
                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data
                
                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i]           = output[i].cpu().numpy()
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = self.correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
                 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

