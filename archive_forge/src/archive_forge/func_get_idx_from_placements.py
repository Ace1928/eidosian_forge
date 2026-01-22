import copy
from typing import List, Tuple
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
import torch.distributed._shard.sharding_spec as shard_spec
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed.nn.functional import (
from torch.distributed._shard.metadata import ShardMetadata
from .shard import Shard
def get_idx_from_placements(placements, current_rank) -> int:
    """
    Return the position of the current rank in the given placements.

    Args:
        placements(List[Union[_remote_device, str]]):
            Specifies the placement of each shard of the Tensor. The size of
            the list represents the number of shards to be created. This could
            be a list of
            :class:`torch.distributed._remote_device`'s. This list
            could also contain a string which represents remote
            device as accepted by
            :class:`torch.distributed._remote_device`
        current_rank (int): number of current device.

    Returns:
        A int which contains the position of current device in the placement list.
    """
    for idx, placement in enumerate(placements):
        if current_rank == placement.rank():
            return idx
    raise RuntimeError('current_rank not in the placement.')