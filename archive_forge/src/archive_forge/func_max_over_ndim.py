import torch
import copy
from typing import Dict, Any
def max_over_ndim(input, axis_list, keepdim=False):
    """Apply 'torch.max' over the given axes."""
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.max(axis, keepdim)
    return input