import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.prod.dim_int, 'self: jt, dim: any, keepdim: any?')
def prod_dim_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop('input')
    if not new_kwargs['keepdim']:
        raise RuntimeError('prod(): keepdim=True must be set for NestedTensor')
    dim = new_kwargs['dim']
    new_kwargs['dim'] = _wrap_jagged_dim(len(inp.shape), dim, 'prod')
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(args[0]))