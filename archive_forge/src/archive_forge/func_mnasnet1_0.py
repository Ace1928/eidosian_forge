import warnings
from functools import partial
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
@register_model()
@handle_legacy_interface(weights=('pretrained', MNASNet1_0_Weights.IMAGENET1K_V1))
def mnasnet1_0(*, weights: Optional[MNASNet1_0_Weights]=None, progress: bool=True, **kwargs: Any) -> MNASNet:
    """MNASNet with depth multiplier of 1.0 from
    `MnasNet: Platform-Aware Neural Architecture Search for Mobile
    <https://arxiv.org/abs/1807.11626>`_ paper.

    Args:
        weights (:class:`~torchvision.models.MNASNet1_0_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MNASNet1_0_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mnasnet.MNASNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MNASNet1_0_Weights
        :members:
    """
    weights = MNASNet1_0_Weights.verify(weights)
    return _mnasnet(1.0, weights, progress, **kwargs)