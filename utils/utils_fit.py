import os
import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, fcos_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, save_period, save_dir):
    total_loss  = 0
    val_loss    = 0
    
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, bboxes, classes = batch[0], batch[1],  batch[2]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    bboxes  = torch.from_numpy(bboxes).type(torch.FloatTensor).cuda()
                    classes = torch.from_numpy(classes).type(torch.LongTensor).cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    bboxes  = torch.from_numpy(bboxes).type(torch.FloatTensor)
                    classes = torch.from_numpy(classes).type(torch.LongTensor)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #-------------------#
            #   获得预测结果
            #-------------------#
            outputs = model_train(images)
            #-------------------#
            #   计算损失
            #-------------------#
            loss = fcos_loss(outputs, bboxes, classes)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(**{'loss'  : total_loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, bboxes, classes = batch[0], batch[1],  batch[2]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    bboxes  = torch.from_numpy(bboxes).type(torch.FloatTensor).cuda()
                    classes = torch.from_numpy(classes).type(torch.LongTensor).cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    bboxes  = torch.from_numpy(bboxes).type(torch.FloatTensor)
                    classes = torch.from_numpy(classes).type(torch.LongTensor)
                    
                optimizer.zero_grad()
                outputs     = model_train(images)
                loss        = fcos_loss(outputs, bboxes, classes)
                val_loss    += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
