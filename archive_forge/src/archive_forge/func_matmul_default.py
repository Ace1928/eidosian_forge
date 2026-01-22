import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.matmul.default, 'self: jt, other: any')
def matmul_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop('input')
    other = new_kwargs.pop('other')
    if inp.is_nested and (not other.is_nested):
        return NestedTensor(func(inp._values, other, **new_kwargs), **extract_kwargs(inp))
    elif inp.is_nested and other.is_nested:
        if inp.dim() > 3 and other.dim() > 3 and raggedness_matches(inp, other.shape):
            return NestedTensor(func(inp._values, other._values), **extract_kwargs(inp))
    raise RuntimeError(f'matmul(): not supported between inputs of shapes {inp.shape} and {other.shape}')