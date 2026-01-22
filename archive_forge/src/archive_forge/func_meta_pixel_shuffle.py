import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
@register_meta(aten.pixel_shuffle.default)
def meta_pixel_shuffle(self, upscale_factor):
    assert len(self.shape) > 2 and self.shape[-3] % (upscale_factor * upscale_factor) == 0, f'Invalid input shape for pixel_shuffle: {self.shape} with upscale_factor = {upscale_factor}'

    def is_channels_last(ten):
        return torch._prims_common.suggest_memory_format(ten) == torch.channels_last

    def pick_memory_format():
        if is_channels_last(self):
            if device_hint(self) == 'cuda':
                return torch.contiguous_format
            else:
                return torch.channels_last
        elif self.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        elif self.is_contiguous(memory_format=torch.preserve_format):
            return torch.preserve_format
    C = self.shape[-3] // (upscale_factor * upscale_factor)
    Hr = self.shape[-2] * upscale_factor
    Wr = self.shape[-1] * upscale_factor
    out_shape = (*self.shape[:-3], C, Hr, Wr)
    out = self.new_empty(out_shape)
    out = out.to(memory_format=pick_memory_format())
    return out