import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.cat.default, 'tensors: any, dim: any')
def cat_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    tensors = new_kwargs.pop('tensors')
    nested = [t for t in tensors if t.is_nested]
    assert len(nested) > 0
    first = nested[0]
    tensors = [t if t.is_nested else t.expand_as(first) for t in tensors]
    dim = new_kwargs['dim']
    new_kwargs['dim'] = _wrap_jagged_dim(len(first.shape), dim, 'cat')
    return NestedTensor(func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0]))