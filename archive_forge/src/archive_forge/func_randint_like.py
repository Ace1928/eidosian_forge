import functools
import logging
import math
import sys
import typing
from typing import Optional
import torch
import torch._decomp as decomp
import torch._prims_common as utils
import torch.ao.quantization.fx._decomposed
from torch._decomp import (
from torch._decomp.decompositions import (
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import type_to_dtype
from . import config, inductor_prims
@register_decomposition(aten.randint_like.default)
def randint_like(self, high, *, dtype=None, device=None, memory_format=None, **kwargs):
    return aten.randint.low(0, high, [*self.size()], dtype=dtype or self.dtype, device=device or self.device, **kwargs).to(memory_format=get_like_layout(self, memory_format))