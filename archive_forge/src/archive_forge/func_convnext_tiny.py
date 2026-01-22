from functools import partial
from typing import Any, Callable, List, Optional, Sequence
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ..ops.misc import Conv2dNormActivation, Permute
from ..ops.stochastic_depth import StochasticDepth
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
@register_model()
@handle_legacy_interface(weights=('pretrained', ConvNeXt_Tiny_Weights.IMAGENET1K_V1))
def convnext_tiny(*, weights: Optional[ConvNeXt_Tiny_Weights]=None, progress: bool=True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Tiny model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Tiny_Weights
        :members:
    """
    weights = ConvNeXt_Tiny_Weights.verify(weights)
    block_setting = [CNBlockConfig(96, 192, 3), CNBlockConfig(192, 384, 3), CNBlockConfig(384, 768, 9), CNBlockConfig(768, None, 3)]
    stochastic_depth_prob = kwargs.pop('stochastic_depth_prob', 0.1)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)