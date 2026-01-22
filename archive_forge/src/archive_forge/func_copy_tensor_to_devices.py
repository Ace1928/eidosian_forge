import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def copy_tensor_to_devices(tensor=None) -> torch.Tensor:
    """
    Copys a tensor that only exists on a single device and broadcasts it to other devices. Differs from `broadcast` as
    each worker doesn't need to know its shape when used (and tensor can be `None`)

    Args:
        tensor (`torch.tensor`):
            The tensor that should be sent to all devices. Must only have it be defined on a single device, the rest
            should be `None`.
    """
    state = PartialState()
    shape, dtype = gather_tensor_shape(tensor)
    if tensor is None:
        tensor = torch.zeros(shape, dtype=TENSOR_INT_TO_DTYPE[dtype]).to(state.device)
    return reduce(tensor, reduction='sum')