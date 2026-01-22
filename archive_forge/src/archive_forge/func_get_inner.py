import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
def get_inner(aten_op):

    def inner(*args, **kwargs):
        check_schema(schema_str, func, *args, **kwargs)
        return func(aten_op, *args, **kwargs)
    return inner