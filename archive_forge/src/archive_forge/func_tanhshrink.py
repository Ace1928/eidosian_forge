import math
from functools import wraps
from typing import Callable, Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _make_inplace
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(type_promoting_args=('a',), type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def tanhshrink(a: TensorLikeType) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.tanhshrink
    """
    if not isinstance(a, TensorLike):
        raise RuntimeError('Expected a tensor input for an elementwise unary operation!')
    return a - torch.tanh(a)