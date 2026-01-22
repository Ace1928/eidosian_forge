import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple
import torch
def reorder_fsms(fsms: List['Guide'], ancestors: torch.Tensor) -> List['Guide']:
    reordered_fsms = []
    for ancestor in ancestors:
        reordered_fsms.append(fsms[ancestor].copy())
    return reordered_fsms