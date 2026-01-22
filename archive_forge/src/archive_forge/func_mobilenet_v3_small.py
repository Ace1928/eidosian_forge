from functools import partial
from typing import Any, Callable, List, Optional, Sequence
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation as SElayer
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
@register_model()
@handle_legacy_interface(weights=('pretrained', MobileNet_V3_Small_Weights.IMAGENET1K_V1))
def mobilenet_v3_small(*, weights: Optional[MobileNet_V3_Small_Weights]=None, progress: bool=True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__.

    Args:
        weights (:class:`~torchvision.models.MobileNet_V3_Small_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V3_Small_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mobilenet.MobileNetV3``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MobileNet_V3_Small_Weights
        :members:
    """
    weights = MobileNet_V3_Small_Weights.verify(weights)
    inverted_residual_setting, last_channel = _mobilenet_v3_conf('mobilenet_v3_small', **kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)