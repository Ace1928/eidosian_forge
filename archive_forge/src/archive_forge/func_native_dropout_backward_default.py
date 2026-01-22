import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.native_dropout_backward.default, 'grad_output: jt, mask: jt, scale: any')
def native_dropout_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    grad_output = new_kwargs.pop('grad_output')
    mask = new_kwargs.pop('mask')
    return NestedTensor(func(grad_output._values, mask._values, **new_kwargs), **extract_kwargs(grad_output))