from typing import List, Optional
import torch
from .utils import ensure_divisibility
def get_data_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP