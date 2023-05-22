from tqdm import tqdm
import models
import utils
import os
import argparse
import numpy as np
from torch.utils import data
from utils.metrics import StreamSegMetrics
import torch
import torch.nn as nn

import warnings

warnings.filterwarnings("ignore")
from models.utils import model_setting, init_bn_momentum
from datasets.builder_dataset import build_dataset
from models.builder_model import build_model
from utils.utils import get_save_dir
from utils.checkpoint import save_ckpt, load_priormodel_ckpt,load_resume_ckpt
from val import validate

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils.optim.builder_optimizer import build_optimizer

from utils.losses.builder_loss import build_loss
from utils.lr_scheduler import build_scheduler


def get_argparser():
    parser = argparse.ArgumentParser()
    ############ set data parameters #########
    parser.add_argument("--data_root", type=str, default=r'E:\project\DATA\cityscapes_data\mydata_vertify',
                        help="path to Dataset")
    # ['voc', 'cityscapes','cityscapes_mydata']
    parser.add_argument("--dataset", type=str, default='cityscapes_mydata', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=3, help="num classes (default: None)")
    # PASCAL VOC Options choices = ['2012_aug', '2012', '2011', '2009', '2008', '2007']
    parser.add_argument("--year", type=str, default='2012', help='year of VOC')

    ################ set model parameters#################

    parser.add_argument("--model", type=str, default='Deeplabv3plus', help='model name')
    parser.add_argument("--backbone", type=str, default='mobilenet_v2', help='backbone name')
    parser.add_argument("--ckpt", default='weights/best_deeplabv3plus_mobilenet_cityscapes_os16.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--load_ckpt", default=True, type=bool, help="decision load ckpt weights")
    parser.add_argument("--resume", default=False, type=bool, help="resume checkpoint")
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    parser.add_argument("--crop_val", action='store_true', default=True, help='crop validation (default: False)')

    parser.add_argument("--batch_size", type=int, default=8, help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4, help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--num_workers", type=int, default=0, help="num classes (default: None)")

    parser.add_argument('--total_epochs', type=int, default=300, help="the number of epochs: 300 for train")
    parser.add_argument('--save_period', type=int, default=20, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--val_period', type=int, default=10, help='val every x epochs (disabled if < 1)')
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--download", action='store_true', default=False, help="download datasets")

    parser.add_argument("--out_dir", type=str, default='./runs/train', help='backbone name')

    # loss optimzer  parameters
    parser.add_argument("--loss", type=str, default='Cross_entropy', help='loss name')
    parser.add_argument('--lr', type=float, default=0.01, help="initial learning rate")
    parser.add_argument('--optim', type=str.lower, default='sgd', choices=['sgd', 'adam', 'adamw'],
                        help="select optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--lr_policy", type=str, default='PolyLR', choices=['poly', 'step'],
                        help="learning rate scheduler policy")

    opts = parser.parse_args()

    if opts.dataset == 'voc':
        opts.num_classes = 21
    elif opts.dataset == 'cityscapes':
        opts.num_classes = 19
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    return opts


def main():
    opts = get_argparser()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Device: %s" % device)
    device = torch.device('cuda:0')

    model_setting(opts)  # 模型配置

    # Setup dataloader
    train_dst, val_dst = build_dataset(opts)
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers,
                                   drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=opts.num_workers)
    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))

    model = build_model(opts.model, opts.num_classes, backbone_name=opts.backbone, out_stride=opts.output_stride,
                        pretrained=False)
    init_bn_momentum(model.backbone, momentum=0.01)
    if opts.separable_conv and 'plus' in opts.model:
        models.convert_to_separable_conv(model.classifier)



    if opts.load_ckpt and not opts.resume:  # 决定是否载入ckpt权重
        model = load_priormodel_ckpt(opts.ckpt, model)
    if opts.resume:
        model = load_resume_ckpt(opts.ckpt, model)


    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    optimizer = build_optimizer(opts.optim, model, opts.lr)
    criterion = build_loss(opts.loss, weight_coefficient=None, ignore_label=255)
    save_dir = get_save_dir(opts.out_dir,opts.resume)
    opts.save_dir = save_dir
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], output_device=[0])
    model = nn.DataParallel(model, device_ids=[0])
    # model.to(device)




    best_miou = 0
    max_iter = opts.total_epochs * len(train_loader)
    scheduler = build_scheduler(opts.lr_policy, optimizer, **{"max_iter": max_iter})

    for epoch in range(opts.total_epochs):
        model.train()
        with tqdm(total=len(train_loader)) as pbar:
            for iter, (images, labels) in enumerate(train_loader):
                images = torch.tensor(images,dtype=torch.float32).cuda()#images.to(device, dtype=torch.float32)
                labels = torch.tensor(labels,dtype=torch.long).cuda() #labels.to(device, dtype=torch.long)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                np_loss = loss.detach().cpu().numpy()
                cur_iter = (epoch * len(train_loader)) + iter + 1

                scheduler.step()

                #################### 打印信息控制############
                if (iter+1) % 100 == 0:
                    print('\tepoch: {}|{}\tloss:{}'.format(epoch + 1, iter + 1, np_loss))
                pbar.set_description("epoch {}|{}".format(opts.total_epochs, epoch + 1))
                pbar.set_postfix(iter_all='{}||{}'.format(max_iter, cur_iter),
                                 iter_epoch='{}||{}'.format(len(train_loader), iter + 1), loss=np_loss)
                pbar.update()

                break

            save_ckpt(os.path.join(save_dir, 'last.pth'), model, optimizer, scheduler, epoch, best_miou)
            if (epoch) % opts.save_period == 0 and opts.save_period > 0:
                pth_name = str(opts.model) + "_" + str(opts.backbone) + '_' + str(epoch+1) + '.pth'
                save_ckpt(os.path.join(save_dir, pth_name), model, optimizer, scheduler, epoch, best_miou)

            if opts.val_period>0 and epoch%opts.val_period!=0:
                continue


            print("\nvalidation...")
            val_score = validate(model=model, loader=val_loader, device=device, metrics=metrics)

            if val_score['Mean IoU'] > best_miou:  # save best model
                best_miou = val_score['Mean IoU']
                save_ckpt(os.path.join(save_dir, 'best.pth'), model, optimizer, scheduler, epoch, best_miou)

            print('\nOverall Acc\t{}\nFreqW Acc\t{}\nMean IoU\t{}\n'.format(val_score['Overall Acc'],
                                                                            val_score['FreqW Acc'],
                                                                            val_score['Mean IoU'],
                                                                            val_score['Class IoU']))




if __name__ == '__main__':
    main()
