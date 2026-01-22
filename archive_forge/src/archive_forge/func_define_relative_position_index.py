from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..swin_transformer import PatchMerging, SwinTransformerBlock
def define_relative_position_index(self) -> None:
    coords_dhw = [torch.arange(self.window_size[i]) for i in range(3)]
    coords = torch.stack(torch.meshgrid(coords_dhw[0], coords_dhw[1], coords_dhw[2], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += self.window_size[0] - 1
    relative_coords[:, :, 1] += self.window_size[1] - 1
    relative_coords[:, :, 2] += self.window_size[2] - 1
    relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
    relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
    relative_position_index = relative_coords.sum(-1)
    self.register_buffer('relative_position_index', relative_position_index)