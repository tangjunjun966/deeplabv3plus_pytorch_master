import os
import argparse
import numpy as np
from datasets import VOCSegmentation, Cityscapes,Cityscapes_mydata

import torch
import torch.nn as nn

from PIL import Image

import matplotlib.pyplot as plt
from models._deeplab import convert_to_separable_conv
from models.builder_model import build_model
from models.utils import init_bn_momentum
from utils.checkpoint import load_ckpt
from datasets.builder_dataset import transform
from tqdm import tqdm
import cv2
from utils.utils import get_save_dir


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='cityscapes_mydata', help='Name of training set')
    parser.add_argument("--model", type=str, default='Deeplabv3plus', help='model name')
    parser.add_argument("--backbone", type=str, default='mobilenet_v2', help='backbone name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--out_dir", default='./runs/detect', help="save segmentation results dir")
    parser.add_argument("--ckpt", default='./weights/best.pth', type=str,
                        help="resume from checkpoint")
    parser.add_argument("--num_classes", type=int, default=3, help="num classes (default: None)")
    return parser


def show_img(img):
    plt.imshow(img)
    plt.show()


opts = get_argparser().parse_args()


def init_model():
    model = build_model(opts.model, opts.num_classes, backbone_name=opts.backbone, out_stride=16,
                        pretrained=False)

    if opts.separable_conv and 'plus' in opts.model:
        convert_to_separable_conv(model.classifier)
    init_bn_momentum(model.backbone, momentum=0.01)

    model = load_ckpt(opts.ckpt, model)

    model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    model = model.eval()

    return model


def infer(img, model):
    with torch.no_grad():
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = transform(img).unsqueeze(0)  # To tensor of NCHW
        img = img.cuda() if torch.cuda.is_available() else img.cpu()
        pred = model(img)  # [b,class_num,h,w]已在维度1上获得class_num个类别的置信度，可使用max获得类别
        pred = pred.max(1)[1].cpu().numpy()[0]  # HW #max= 第一个是值，按dim=0比较得到的 # 第二个是值对应的索引，对应维度dim=0上的索引
    return pred


def demo(root_data):
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == 'cityscapes_mydata':
        decode_fn = Cityscapes_mydata.decode_target

    model = init_model()
    save_dir = get_save_dir(opts.out_dir)
    for img_name in tqdm(os.listdir(root_data)):
        if img_name[-3:] in ['png', 'jpg']:
            img_path = os.path.join(root_data, img_name)
            img = cv2.imread(img_path)
            pred = infer(img, model)
            colorized_preds = decode_fn(pred).astype('uint8')

            img_merge = cv2.addWeighted(img, 0.6, colorized_preds, 0.4, 0)
            img_save = np.concatenate((np.array(img_merge), np.array(colorized_preds)), axis=0)

            cv2.imwrite(os.path.join(save_dir, img_name), img_save)


if __name__ == '__main__':
    # root_data = r'E:\project\DATA\cityscapes_data\cityscapes\leftImg8bit\test\berlin'
    root_data = r'C:\Users\Administrator\Desktop\deeplabv3\imgs'
    # root_data=r'E:\project\DATA\cityscapes_data\mydata_vertify\images\val'
    demo(root_data)
