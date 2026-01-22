import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def ignorant_find_batch_size(data):
    """
    Same as [`utils.operations.find_batch_size`] except will ignore if `ValueError` and `TypeErrors` are raised

    Args:
        data (nested list/tuple/dictionary of `torch.Tensor`): The data from which to find the batch size.

    Returns:
        `int`: The batch size.
    """
    try:
        return find_batch_size(data)
    except (ValueError, TypeError):
        pass
    return None