import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
@register_model()
@handle_legacy_interface(weights=('pretrained', EfficientNet_B0_Weights.IMAGENET1K_V1))
def efficientnet_b0(*, weights: Optional[EfficientNet_B0_Weights]=None, progress: bool=True, **kwargs: Any) -> EfficientNet:
    """EfficientNet B0 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B0_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B0_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_B0_Weights
        :members:
    """
    weights = EfficientNet_B0_Weights.verify(weights)
    inverted_residual_setting, last_channel = _efficientnet_conf('efficientnet_b0', width_mult=1.0, depth_mult=1.0)
    return _efficientnet(inverted_residual_setting, kwargs.pop('dropout', 0.2), last_channel, weights, progress, **kwargs)