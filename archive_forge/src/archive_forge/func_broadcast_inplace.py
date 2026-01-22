import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def broadcast_inplace(a: List[int], b: List[int]):
    dimsA = len(a)
    dimsB = len(b)
    if dimsB > dimsA:
        raise AssertionError(f'The dims of tensor b ({dimsB}) must be less than or equal tothe dims of tensor a ({dimsA}) ')
    for dimA in range(dimsA):
        dimB = dimsB - dimsA + dimA
        sizeA = a[dimA]
        sizeB = b[dimB] if dimB >= 0 else 1
        if sizeA != sizeB and sizeB != 1:
            raise AssertionError('The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}'.format(sizeA, sizeB, dimA))
    return _copy(a)