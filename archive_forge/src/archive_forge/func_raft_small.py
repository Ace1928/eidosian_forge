from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.ops import Conv2dNormActivation
from ...transforms._presets import OpticalFlow
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._utils import handle_legacy_interface
from ._utils import grid_sample, make_coords_grid, upsample_flow
@register_model()
@handle_legacy_interface(weights=('pretrained', Raft_Small_Weights.C_T_V2))
def raft_small(*, weights: Optional[Raft_Small_Weights]=None, progress=True, **kwargs) -> RAFT:
    """RAFT "small" model from
    `RAFT: Recurrent All Pairs Field Transforms for Optical Flow <https://arxiv.org/abs/2003.12039>`__.

    Please see the example below for a tutorial on how to use this model.

    Args:
        weights(:class:`~torchvision.models.optical_flow.Raft_Small_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.optical_flow.Raft_Small_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.optical_flow.RAFT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.optical_flow.Raft_Small_Weights
        :members:
    """
    weights = Raft_Small_Weights.verify(weights)
    return _raft(weights=weights, progress=progress, feature_encoder_layers=(32, 32, 64, 96, 128), feature_encoder_block=BottleneckBlock, feature_encoder_norm_layer=InstanceNorm2d, context_encoder_layers=(32, 32, 64, 96, 160), context_encoder_block=BottleneckBlock, context_encoder_norm_layer=None, corr_block_num_levels=4, corr_block_radius=3, motion_encoder_corr_layers=(96,), motion_encoder_flow_layers=(64, 32), motion_encoder_out_channels=82, recurrent_block_hidden_state_size=96, recurrent_block_kernel_size=(3,), recurrent_block_padding=(1,), flow_head_hidden_size=128, use_mask_predictor=False, **kwargs)