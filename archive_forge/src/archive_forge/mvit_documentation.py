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

        MViT main class.

        Args:
            spatial_size (tuple of ints): The spacial size of the input as ``(H, W)``.
            temporal_size (int): The temporal size ``T`` of the input.
            block_setting (sequence of MSBlockConfig): The Network structure.
            residual_pool (bool): If True, use MViTv2 pooling residual connection.
            residual_with_cls_embed (bool): If True, the addition on the residual connection will include
                the class embedding.
            rel_pos_embed (bool): If True, use MViTv2's relative positional embeddings.
            proj_after_attn (bool): If True, apply the projection after the attention.
            dropout (float): Dropout rate. Default: 0.0.
            attention_dropout (float): Attention dropout rate. Default: 0.0.
            stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
            num_classes (int): The number of classes.
            block (callable, optional): Module specifying the layer which consists of the attention and mlp.
            norm_layer (callable, optional): Module specifying the normalization layer to use.
            patch_embed_kernel (tuple of ints): The kernel of the convolution that patchifies the input.
            patch_embed_stride (tuple of ints): The stride of the convolution that patchifies the input.
            patch_embed_padding (tuple of ints): The padding of the convolution that patchifies the input.
        