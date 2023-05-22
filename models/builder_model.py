import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch.nn as nn

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',


}

from .DeeplabV3Plus import Deeplabv3plus


def build_model(model_name, num_classes, backbone_name='mobilenet_v2', out_stride=32, pretrained=False, mult_grid=False):

    model=None
    if model_name == 'Deeplabv3plus':
        model = Deeplabv3plus(backbone_name, num_classes=num_classes, out_stride=out_stride, pretrained=pretrained)


    # elif model_name == 'PSANet50':
    #     return PSANet(layers=50, dropout=0.1, classes=num_classes, zoom_factor=8, use_psa=True, psa_type=2, compact=compact,
    #                shrink_factor=shrink_factor, mask_h=mask_h, mask_w=mask_w, psa_softmax=True, pretrained=True)

    # if pretrained:
    #     checkpoint = model_zoo.load_url(model_urls[backbone])
    #     model_dict = model.state_dict()
    #     # print(model_dict)
    #     # Screen out layers that are not loaded
    #     pretrained_dict = {'backbone.' + k: v for k, v in checkpoint.items() if 'backbone.' + k in model_dict}
    #     # Update the structure dictionary for the current models
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)

    return model






