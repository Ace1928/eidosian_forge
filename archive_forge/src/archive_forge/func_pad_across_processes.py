import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
@chained_operation
def pad_across_processes(tensor, dim=0, pad_index=0, pad_first=False):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so they
    can safely be gathered.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to gather.
        dim (`int`, *optional*, defaults to 0):
            The dimension on which to pad.
        pad_index (`int`, *optional*, defaults to 0):
            The value with which to pad.
        pad_first (`bool`, *optional*, defaults to `False`):
            Whether to pad at the beginning or the end.
    """

    def _pad_across_processes(tensor, dim=0, pad_index=0, pad_first=False):
        if getattr(tensor, 'is_nested', False):
            warnings.warn('Cannot pad nested tensors without more information. Leaving unprocessed.', CannotPadNestedTensorWarning)
            return tensor
        if dim >= len(tensor.shape):
            return tensor
        size = torch.tensor(tensor.shape, device=tensor.device)[None]
        sizes = gather(size).cpu()
        max_size = max((s[dim] for s in sizes))
        if max_size == tensor.shape[dim]:
            return tensor
        old_size = tensor.shape
        new_size = list(old_size)
        new_size[dim] = max_size
        new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
        if pad_first:
            indices = tuple((slice(max_size - old_size[dim], max_size) if i == dim else slice(None) for i in range(len(new_size))))
        else:
            indices = tuple((slice(0, old_size[dim]) if i == dim else slice(None) for i in range(len(new_size))))
        new_tensor[indices] = tensor
        return new_tensor
    return recursively_apply(_pad_across_processes, tensor, error_on_other_type=True, dim=dim, pad_index=pad_index, pad_first=pad_first)