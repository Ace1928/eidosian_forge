from functools import partial
from typing import Any, Callable, Optional
import torch
from torch import nn
from torchvision.ops.misc import Conv3dNormActivation
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
Construct Separable 3D CNN model.

    Reference: `Rethinking Spatiotemporal Feature Learning <https://arxiv.org/abs/1712.04851>`__.

    .. betastatus:: video module

    Args:
        weights (:class:`~torchvision.models.video.S3D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.S3D_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.S3D`` base class.
            Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/s3d.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.S3D_Weights
        :members:
    