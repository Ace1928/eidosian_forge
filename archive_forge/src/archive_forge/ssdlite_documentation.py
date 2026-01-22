import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch import nn, Tensor
from ...ops.misc import Conv2dNormActivation
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .. import mobilenet
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from . import _utils as det_utils
from .anchor_utils import DefaultBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .ssd import SSD, SSDScoringHead
SSDlite model architecture with input size 320x320 and a MobileNetV3 Large backbone, as
    described at `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__ and
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks <https://arxiv.org/abs/1801.04381>`__.

    .. betastatus:: detection module

    See :func:`~torchvision.models.detection.ssd300_vgg16` for more details.

    Example:

        >>> model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        >>> model.eval()
        >>> x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        weights (:class:`~torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model
            (including the background).
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained
            weights for the backbone.
        trainable_backbone_layers (int, optional): number of trainable (not frozen) layers
            starting from final block. Valid values are between 0 and 6, with 6 meaning all
            backbone layers are trainable. If ``None`` is passed (the default) this value is
            set to 6.
        norm_layer (callable, optional): Module specifying the normalization layer to use.
        **kwargs: parameters passed to the ``torchvision.models.detection.ssd.SSD``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights
        :members:
    