# _*_ coding: utf-8 _*_
"""
Time:     2023/5/17 17:02
Author:   jun tang(owen)
File:     builder_loss.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import torch
from utils.losses.loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, \
    ProbOhemCrossEntropy2d, FocalLoss2d, LabelSmoothing


def build_loss(loss_name, weight_coefficient=None, ignore_label=255,**kwargs):


    weight = None if weight_coefficient is  None  else torch.from_numpy(weight_coefficient)
    # Default uses cross quotient loss function
    criteria = None
    if loss_name == 'Cross_entropy':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label, reduction='mean')
    elif loss_name == 'ProbOhemCrossEntropy2d':
        min_kept=kwargs.get('min_kept')
        if min_kept is None:
            raise ValueError("min_kept no value, computational formula=batch_size // len(gpus_id) * img_h * img_w // 16 ")

        criteria = ProbOhemCrossEntropy2d(weight=weight, ignore_label=ignore_label, thresh=0.7, min_kept=min_kept)
    elif loss_name == 'CrossEntropyLoss2dLabelSmooth':
        criteria = CrossEntropyLoss2dLabelSmooth(weight=weight, ignore_label=ignore_label)
    elif loss_name == 'LovaszSoftmax':
        criteria = LovaszSoftmax(ignore_index=ignore_label)
    elif loss_name == 'FocalLoss':
        criteria = FocalLoss2d(weight=weight, ignore_index=ignore_label)

    return criteria
