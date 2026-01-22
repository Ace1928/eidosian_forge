import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.is_contiguous.default, 'self: jt_all')
def is_contiguous_general(func, *args, **kwargs):
    from torch._prims_common import is_contiguous_for_memory_format
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop('input')
    if inp.lengths() is not None:
        return False
    if inp._ragged_idx != 1:
        return False
    new_kwargs['memory_format'] = new_kwargs.get('memory_format', torch.contiguous_format)
    if new_kwargs['memory_format'] == torch.preserve_format:
        return True
    return is_contiguous_for_memory_format(inp.values(), **new_kwargs)