from contextlib import contextmanager
from datetime import timedelta
from functools import (
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
def with_fake_comms(func=None, rank=0, world_size=2):
    """
    Function wrapper that inits a fake process group designed for testing.
    Right now only querying for world size is available
    """
    if func is None:
        return partial(with_fake_comms, rank=rank, world_size=world_size)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with with_dist(rank, world_size):
            func(self, *args, **kwargs)
    return wrapper