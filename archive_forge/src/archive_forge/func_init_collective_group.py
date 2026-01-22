import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def init_collective_group(world_size: int, rank: int, backend=types.Backend.NCCL, group_name: str='default'):
    """Initialize a collective group inside an actor process.

    Args:
        world_size: the total number of processes in the group.
        rank: the rank of the current process.
        backend: the CCL backend to use, NCCL or GLOO.
        group_name: the name of the collective group.

    Returns:
        None
    """
    _check_inside_actor()
    backend = types.Backend(backend)
    _check_backend_availability(backend)
    global _group_mgr
    if not group_name:
        raise ValueError("group_name '{}' needs to be a string.".format(group_name))
    if _group_mgr.is_group_exist(group_name):
        raise RuntimeError('Trying to initialize a group twice.')
    assert world_size > 0
    assert rank >= 0
    assert rank < world_size
    _group_mgr.create_collective_group(backend, world_size, rank, group_name)