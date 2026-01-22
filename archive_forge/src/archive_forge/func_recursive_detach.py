import gc
from typing import Any
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
def recursive_detach(in_dict: Any, to_cpu: bool=False) -> Any:
    """Detach all tensors in `in_dict`.

    May operate recursively if some of the values in `in_dict` are dictionaries
    which contain instances of `Tensor`. Other types in `in_dict` are
    not affected by this utility function.

    Args:
        in_dict: Dictionary with tensors to detach
        to_cpu: Whether to move tensor to cpu

    Return:
        out_dict: Dictionary with detached tensors

    """

    def detach_and_move(t: Tensor, to_cpu: bool) -> Tensor:
        t = t.detach()
        if to_cpu:
            t = t.cpu()
        return t
    return apply_to_collection(in_dict, Tensor, detach_and_move, to_cpu=to_cpu)