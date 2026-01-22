import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def gather_tensor_shape(tensor):
    """
    Grabs the shape of `tensor` only available on one process and returns a tensor of its shape
    """
    max_tensor_dimension = 2 ** 20
    state = PartialState()
    base_tensor = torch.empty(max_tensor_dimension, dtype=torch.int, device=state.device)
    if tensor is not None:
        shape = tensor.shape
        tensor_dtype = TENSOR_TYPE_TO_INT[tensor.dtype]
        base_tensor[:len(shape) + 1] = torch.tensor(list(shape) + [tensor_dtype], dtype=int)
    base_tensor = reduce(base_tensor, reduction='sum')
    base_tensor = base_tensor[base_tensor.nonzero()]
    dtype = int(base_tensor[-1:][0])
    base_tensor = base_tensor[:-1]
    return (base_tensor, dtype)