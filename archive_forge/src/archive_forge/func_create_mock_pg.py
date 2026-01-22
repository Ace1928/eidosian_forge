from contextlib import contextmanager
from datetime import timedelta
from functools import (
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
def create_mock_pg(prefix_store, rank, world_size, timeout):
    return MockProcessGroup(rank, world_size)