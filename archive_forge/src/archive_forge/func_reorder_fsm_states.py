import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple
import torch
def reorder_fsm_states(fsm_states: List[int], ancestors: torch.Tensor) -> List[int]:
    reordered_states = []
    for ancestor in ancestors:
        reordered_states.append(fsm_states[ancestor])
    return reordered_states