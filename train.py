#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from nets.fcos import FCOS
from nets.fcos_training import Fcos_Loss, get_lr_scheduler, set_optimizer_lr
from utils.callbacks import LossHistory
from utils.dataloader import FcosDataset, fcos_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    #---------------------------------#
    Cuda = True
    #---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关 
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #---------------------------------------------------------------------#
    #   strides         构建特征点时的步长，一般不修改。
    #---------------------------------------------------------------------#
    strides         = [8, 16, 32, 64, 128]
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，下面的pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，下面的pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，下面的pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   从0开始训练有两个方案：
    #   1、得益于Mosaic数据增强方法强大的数据增强能力，将UnFreeze_Epoch设置的较大（300及以上）、batch较大（16及以上）、数据较多（万以上）的情况下，
    #      可以设置mosaic=True，直接随机初始化参数开始训练，但得到的效果仍然不如有预训练的情况。（像COCO这样的大数据集可以这样做）
    #   2、了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/Fcos_weights_voc.pth'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #                   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #                   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #                   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = True
    #------------------------------------------------------#
    #   input_shape     输入的shape大小
    #------------------------------------------------------#
    input_shape     = [640, 640]
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练。
    #      
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练： 
    #       Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True（默认参数）
    #       Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False（不冻结训练）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。optimizer_type = 'sgd'，Init_lr = 1e-2。
    #   （二）从主干网络的预训练权重开始训练：
    #       Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 300，Freeze_Train = True（冻结训练）
    #       Init_Epoch = 0，UnFreeze_Epoch = 300，Freeze_Train = False（不冻结训练）
    #       其中：由于从主干网络的预训练权重开始训练，主干的权值不一定适合目标检测，需要更多的训练跳出局部最优解。
    #             UnFreeze_Epoch可以在200-300之间调整，YOLOV5和YOLOX均推荐使用300。optimizer_type = 'sgd'，Init_lr = 1e-2。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    #                   如果设置Freeze_Train=False，建议使用优化器为sgd
    #------------------------------------------------------------------#
    Freeze_Train        = True
    
    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    #------------------------------------------------------------------#
    save_period         = 1
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #------------------------------------------------------------------#
    num_workers         = 1

    #------------------------------------------------------#
    #   train_annotation_path   训练图片路径和标签
    #   val_annotation_path     验证图片路径和标签
    #------------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    class_names, num_classes = get_classes(classes_path)

    model = FCOS(num_classes, pretrained = pretrained)
    if model_path != '':
        #------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    loss_history    = LossHistory("logs/", model, input_shape=input_shape)
    fcos_loss       = Fcos_Loss(strides)
    
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   判断当前batch_size与64的差别，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs         = 64
        Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
        Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset   = FcosDataset(train_lines, (input_shape[0], input_shape[1]))
        val_dataset     = FcosDataset(val_lines, (input_shape[0], input_shape[1]))
        gen     = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=fcos_dataset_collate)
        gen_val = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=fcos_dataset_collate)
                     
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   判断当前batch_size与64的差别，自适应调整学习率
                #-------------------------------------------------------------------#
                nbs         = 64
                Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
                Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen     = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=fcos_dataset_collate)
                gen_val = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=fcos_dataset_collate)

                UnFreeze_flag = True
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, fcos_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, save_period, save_dir)
            
        loss_history.writer.close()
