from typing import List, Optional
import torch
from .utils import ensure_divisibility
def get_model_parallel_rank() -> int:
    """Return my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_model_parallel_group())