


from ._deeplab import DeepLabHead,DeepLabHeadV3Plus,DeepLabV3
from models.utils import IntermediateLayerGetter
from .builder_backbone import build_backbone




def Deeplabv3plus(backbone_name,num_classes, out_stride=32, pretrained=False):
    if out_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone=build_backbone(backbone_name,out_stride,pretrained)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model














