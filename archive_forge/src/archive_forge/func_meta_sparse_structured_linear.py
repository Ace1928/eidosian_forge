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
@register_meta(aten._sparse_semi_structured_linear)
def meta_sparse_structured_linear(input: Tensor, weight: Tensor, _meta: Tensor, bias: Optional[Tensor]=None, _activation_opt: Optional[str]=None):
    output_sizes = list(input.shape)
    if bias is not None:
        assert weight.size(0) == bias.size(0), 'output size mismatch'
    assert weight.size(1) == input.size(-1) / 2
    output_sizes[-1] = weight.size(0)
    assert len(input.shape) == 2, 'we can only handle the squashed input case'
    transposed_strides = (1, input.size(0))
    output = input.new_empty(output_sizes, dtype=input.dtype if input.dtype != torch.int8 else torch.int32).as_strided(output_sizes, transposed_strides)
    return output