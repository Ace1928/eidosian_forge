from contextlib import contextmanager
from datetime import timedelta
from functools import (
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
@contextmanager
def with_dist(rank=0, world_size=2):
    """
    Context manager that initializer c10d with a fake process group.
    """
    mock_init_dist(rank=rank, world_size=world_size)
    try:
        yield
    finally:
        dist.destroy_process_group()