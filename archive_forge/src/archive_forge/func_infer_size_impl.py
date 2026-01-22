import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def infer_size_impl(shape: List[int], numel: int) -> List[int]:
    newsize = 1
    infer_dim: Optional[int] = None
    for dim in range(len(shape)):
        if shape[dim] == -1:
            if infer_dim is not None:
                raise AssertionError('only one dimension can be inferred')
            infer_dim = dim
        elif shape[dim] >= 0:
            newsize *= shape[dim]
        else:
            raise AssertionError('invalid shape dimensions')
    if not (numel == newsize or (infer_dim is not None and newsize > 0 and (numel % newsize == 0))):
        raise AssertionError('invalid shape')
    out = _copy(shape)
    if infer_dim is not None:
        out[infer_dim] = numel // newsize
    return out