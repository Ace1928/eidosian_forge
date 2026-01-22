from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
import torch.nn as nn
from torch import Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .._utils import _ModelURLs
@register_model()
@handle_legacy_interface(weights=('pretrained', R2Plus1D_18_Weights.KINETICS400_V1))
def r2plus1d_18(*, weights: Optional[R2Plus1D_18_Weights]=None, progress: bool=True, **kwargs: Any) -> VideoResNet:
    """Construct 18 layer deep R(2+1)D network as in

    .. betastatus:: video module

    Reference: `A Closer Look at Spatiotemporal Convolutions for Action Recognition <https://arxiv.org/abs/1711.11248>`__.

    Args:
        weights (:class:`~torchvision.models.video.R2Plus1D_18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.R2Plus1D_18_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.resnet.VideoResNet`` base class.
            Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.R2Plus1D_18_Weights
        :members:
    """
    weights = R2Plus1D_18_Weights.verify(weights)
    return _video_resnet(BasicBlock, [Conv2Plus1D] * 4, [2, 2, 2, 2], R2Plus1dStem, weights, progress, **kwargs)