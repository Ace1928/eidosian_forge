import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.view.default, 'self: jt, size: any')
def view_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop('input')
    size = new_kwargs.pop('size')
    if len(size) < 3 or not raggedness_matches(inp, size):
        raise RuntimeError(f'view(): cannot view shape {inp.shape} as {size}')
    jagged_size = [inp._values.shape[0]] + size[2:]
    return NestedTensor(func(inp._values, jagged_size), **extract_kwargs(inp))