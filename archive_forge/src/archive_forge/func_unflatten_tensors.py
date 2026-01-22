import collections
import logging
import sys
from typing import Any, Dict, List, MutableMapping, Set, Tuple
import torch
import torch.distributed as dist
def unflatten_tensors(flat: torch.Tensor, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Args:
        flat (Tensor): flattened dense tensors to unflatten
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return outputs