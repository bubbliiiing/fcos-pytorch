import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet50 import resnet50


class FPN(nn.Module):
    def __init__(self,features=256):
        super(FPN,self).__init__()
        self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
        self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)

        self.conv_3 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_5 =nn.Conv2d(features, features, kernel_size=3, padding=1)

        self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)

        self.apply(self.init_conv_kaiming)
        
    def upsamplelike(self,inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),mode='nearest')
    
    def init_conv_kaiming(self,module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self,x):
        C3, C4, C5 = x
        #-------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #   40, 40, 1024 -> 40, 40, 256
        #   20, 20, 2048 -> 20, 20, 256
        #-------------------------------------#
        P3 = self.prj_3(C3)
        P4 = self.prj_4(C4)
        P5 = self.prj_5(C5)
            
        #------------------------------------------------#
        #   20, 20, 256 -> 40, 40, 256 -> 40, 40, 256
        #------------------------------------------------#
        P4 = P4 + self.upsamplelike([P5, C4])
        #------------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256 -> 80, 80, 256
        #------------------------------------------------#
        P3 = P3 + self.upsamplelike([P4, C3])

        # 80, 80, 256
        P3 = self.conv_3(P3)
        # 40, 40, 256
        P4 = self.conv_4(P4)
        # 20, 20, 256
        P5 = self.conv_5(P5)

        # 10, 10, 256
        P6 = self.conv_out6(P5)
        # 5, 5, 256
        P7 = self.conv_out7(F.relu(P6))
        return [P3,P4,P5,P6,P7]

class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp,self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))
    def forward(self,x):
        return torch.exp(x*self.scale)

class Fcos_Head(nn.Module):
    def __init__(self, in_channel ,num_classes):
        super(Fcos_Head,self).__init__()
        self.num_classes=num_classes
        cls_branch=[]
        reg_branch=[]

        for _ in range(4):
            cls_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True))
            cls_branch.append(nn.GroupNorm(32, in_channel)),
            cls_branch.append(nn.ReLU(True))

            reg_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True))
            reg_branch.append(nn.GroupNorm(32, in_channel)),
            reg_branch.append(nn.ReLU(True))

        self.cls_conv=nn.Sequential(*cls_branch)
        self.reg_conv=nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv2d(in_channel, num_classes, kernel_size=3, padding=1)
        self.cnt_logits = nn.Conv2d(in_channel, 1, kernel_size=3, padding=1)
        self.reg_pred   = nn.Conv2d(in_channel, 4, kernel_size=3, padding=1)
        
        self.apply(self.init_conv_RandomNormal)
        prior = 0.01
        nn.init.constant_(self.cls_logits.bias,-math.log((1 - prior) / prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1) for _ in range(5)])
    
    def init_conv_RandomNormal(self,module,std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,inputs):
        cls_logits  = []
        cnt_logits  = []
        reg_preds   = []
        for index, P in enumerate(inputs):
            cls_conv_out=self.cls_conv(P)
            reg_conv_out=self.reg_conv(P)

            cls_logits.append(self.cls_logits(cls_conv_out))
            cnt_logits.append(self.cnt_logits(cls_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits, cnt_logits, reg_preds


class FCOS(nn.Module):
    def __init__(self, num_classes, fpn_out_channels=256, pretrained=False):
        super().__init__()
        self.backbone   = resnet50(pretrained = pretrained)
        self.fpn        = FPN(fpn_out_channels)
        self.head       = Fcos_Head(fpn_out_channels, num_classes)

    def forward(self,x):
        #-------------------------------------#
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 2048
        #-------------------------------------#
        C3, C4, C5          = self.backbone(x)
        
        #-------------------------------------#
        #   80, 80, 256
        #   40, 40, 256
        #   20, 20, 256
        #   10, 10, 256
        #   5, 5, 256
        #-------------------------------------#
        P3, P4, P5, P6, P7  = self.fpn.forward([C3, C4, C5])
        
        cls_logits, cnt_logits, reg_preds = self.head.forward([P3, P4, P5, P6, P7])
        return [cls_logits, cnt_logits, reg_preds]