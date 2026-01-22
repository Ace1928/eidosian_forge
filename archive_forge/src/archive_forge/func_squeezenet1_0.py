from functools import partial
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.init as init
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
@register_model()
@handle_legacy_interface(weights=('pretrained', SqueezeNet1_0_Weights.IMAGENET1K_V1))
def squeezenet1_0(*, weights: Optional[SqueezeNet1_0_Weights]=None, progress: bool=True, **kwargs: Any) -> SqueezeNet:
    """SqueezeNet model architecture from the `SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        weights (:class:`~torchvision.models.SqueezeNet1_0_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.SqueezeNet1_0_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.squeezenet.SqueezeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.SqueezeNet1_0_Weights
        :members:
    """
    weights = SqueezeNet1_0_Weights.verify(weights)
    return _squeezenet('1_0', weights, progress, **kwargs)