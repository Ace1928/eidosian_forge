import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import torch
import torch.fx
import torch.nn as nn
from ...ops import MLP, StochasticDepth
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
@register_model()
@handle_legacy_interface(weights=('pretrained', MViT_V2_S_Weights.KINETICS400_V1))
def mvit_v2_s(*, weights: Optional[MViT_V2_S_Weights]=None, progress: bool=True, **kwargs: Any) -> MViT:
    """Constructs a small MViTV2 architecture from
    `Multiscale Vision Transformers <https://arxiv.org/abs/2104.11227>`__ and
    `MViTv2: Improved Multiscale Vision Transformers for Classification
    and Detection <https://arxiv.org/abs/2112.01526>`__.

    .. betastatus:: video module

    Args:
        weights (:class:`~torchvision.models.video.MViT_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MViT_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.MViT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MViT_V2_S_Weights
            :members:
    """
    weights = MViT_V2_S_Weights.verify(weights)
    config: Dict[str, List] = {'num_heads': [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8], 'input_channels': [96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768], 'output_channels': [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768], 'kernel_q': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 'kernel_kv': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 'stride_q': [[1, 1, 1], [1, 2, 2], [1, 1, 1], [1, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 2, 2], [1, 1, 1]], 'stride_kv': [[1, 8, 8], [1, 4, 4], [1, 4, 4], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 1, 1], [1, 1, 1]]}
    block_setting = []
    for i in range(len(config['num_heads'])):
        block_setting.append(MSBlockConfig(num_heads=config['num_heads'][i], input_channels=config['input_channels'][i], output_channels=config['output_channels'][i], kernel_q=config['kernel_q'][i], kernel_kv=config['kernel_kv'][i], stride_q=config['stride_q'][i], stride_kv=config['stride_kv'][i]))
    return _mvit(spatial_size=(224, 224), temporal_size=16, block_setting=block_setting, residual_pool=True, residual_with_cls_embed=False, rel_pos_embed=True, proj_after_attn=True, stochastic_depth_prob=kwargs.pop('stochastic_depth_prob', 0.2), weights=weights, progress=progress, **kwargs)