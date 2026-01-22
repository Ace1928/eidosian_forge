import contextlib
from typing import Dict, Iterator, Set, Union
import torch
from torch.cuda import _lazy_call
from torch.utils.checkpoint import detach_variable
from .initialize import get_data_parallel_rank, get_model_parallel_rank
def set_states(self, states: Dict[str, torch.ByteTensor]) -> None:
    """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
    self.states_ = states