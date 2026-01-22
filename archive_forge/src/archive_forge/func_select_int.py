import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.select.int, 'self: jt, dim: any, index: any')
def select_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop('input')
    new_kwargs['dim'] = _wrap_jagged_dim(inp.dim(), new_kwargs['dim'], 'select')
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))