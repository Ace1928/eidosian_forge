import functools
import operator
from typing import cast, Iterable, List, Sequence, Tuple, Union
import torch
from torch.distributed._tensor._collective_utils import redistribute_cost
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.op_schema import OpStrategy
from torch.distributed._tensor.placement_types import (
def is_tensor_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """Check if the shape is shardable according to the spec."""
    shards_map = [1] * len(shape)
    for i, placement in enumerate(spec.placements):
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            shards_map[shard_dim] *= spec.mesh.size(i)
    for i, dim_size in enumerate(shape):
        if dim_size < shards_map[i]:
            return False
    return True