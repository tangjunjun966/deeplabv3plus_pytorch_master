from .backbone import mobilenetv2


def build_backbone(backbone_name='mobilenet_v2', out_stride=32, pretrained=False):
    backbone = None
    if backbone_name == 'mobilenet_v2':
        backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained, output_stride=out_stride)

    else:
        print('build backbone failed')







    return backbone
