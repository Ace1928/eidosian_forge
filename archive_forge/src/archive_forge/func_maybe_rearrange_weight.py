import copy
from typing import Any, Dict, Optional, TypeVar, Union, overload
import warnings
import torch
from torch import Tensor, device, dtype, nn
import torch.nn.functional as F
import bitsandbytes as bnb
from bitsandbytes.autograd._functions import get_tile_inds, undo_layout
from bitsandbytes.functional import QuantState
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.utils import OutlierTracer
def maybe_rearrange_weight(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    weight = state_dict.get(f'{prefix}weight')
    if weight is None:
        return
    weight_format = state_dict.pop(f'{prefix}weight_format', 'row')
    if weight_format != 'row':
        tile_indices = get_tile_inds(weight_format, weight.device)
        state_dict[f'{prefix}weight'] = undo_layout(weight, tile_indices)