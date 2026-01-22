from typing import List, Optional
import torch
from .utils import ensure_divisibility
def get_data_parallel_world_size() -> int:
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())