from functools import partial
from typing import Any, Callable, Dict, List, Type, TypeVar, Union, overload
import torch
import torch.nn as nn
import torch.types
def batched_gather(data: torch.Tensor, inds: torch.Tensor, dim: int=0, no_batch_dims: int=0) -> torch.Tensor:
    ranges: List[Union[slice, torch.Tensor]] = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*(1,) * i, -1, *(1,) * (len(inds.shape) - i - 1)))
        ranges.append(r)
    remaining_dims: List[Union[slice, torch.Tensor]] = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[tuple(ranges)]