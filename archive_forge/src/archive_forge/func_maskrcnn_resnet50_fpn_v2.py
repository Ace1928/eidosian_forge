from collections import OrderedDict
from typing import Any, Callable, Optional
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from ...ops import misc as misc_nn_ops
from ...transforms._presets import ObjectDetection
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..resnet import resnet50, ResNet50_Weights
from ._utils import overwrite_eps
from .backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from .faster_rcnn import _default_anchorgen, FasterRCNN, FastRCNNConvFCHead, RPNHead
@register_model()
@handle_legacy_interface(weights=('pretrained', MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1), weights_backbone=('pretrained_backbone', ResNet50_Weights.IMAGENET1K_V1))
def maskrcnn_resnet50_fpn_v2(*, weights: Optional[MaskRCNN_ResNet50_FPN_V2_Weights]=None, progress: bool=True, num_classes: Optional[int]=None, weights_backbone: Optional[ResNet50_Weights]=None, trainable_backbone_layers: Optional[int]=None, **kwargs: Any) -> MaskRCNN:
    """Improved Mask R-CNN model with a ResNet-50-FPN backbone from the `Benchmarking Detection Transfer
    Learning with Vision Transformers <https://arxiv.org/abs/2111.11429>`_ paper.

    .. betastatus:: detection module

    :func:`~torchvision.models.detection.maskrcnn_resnet50_fpn` for more details.

    Args:
        weights (:class:`~torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers starting from
            final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are
            trainable. If ``None`` is passed (the default) this value is set to 3.
        **kwargs: parameters passed to the ``torchvision.models.detection.mask_rcnn.MaskRCNN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights
        :members:
    """
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)
    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param('num_classes', num_classes, len(weights.meta['categories']))
    elif num_classes is None:
        num_classes = 91
    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    backbone = resnet50(weights=weights_backbone, progress=progress)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, norm_layer=nn.BatchNorm2d)
    rpn_anchor_generator = _default_anchorgen()
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead((backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d)
    mask_head = MaskRCNNHeads(backbone.out_channels, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d)
    model = MaskRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=rpn_anchor_generator, rpn_head=rpn_head, box_head=box_head, mask_head=mask_head, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model