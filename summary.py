import torch
from torchsummary import summary

from nets.fcos import FCOS

if __name__ == "__main__":
    #-------------------------------------------------------#
    #   需要使用device来指定网络在GPU还是CPU运行
    #-------------------------------------------------------#
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = FCOS(80).to(device)
    summary(model, input_size=(3, 800, 800))
