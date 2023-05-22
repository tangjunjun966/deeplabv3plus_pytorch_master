from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os 

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_save_dir(out_dir,resume=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    max_i=0
    for name in os.listdir(out_dir):
       N=len(name)
       if 'exp' in name and N>3:
           v=int(name[3:])
           if v>max_i:  max_i=v
    name='exp'+str(max_i+1)  if not resume else 'exp'+str(max_i)
    save_dir=os.path.join(out_dir,name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir









