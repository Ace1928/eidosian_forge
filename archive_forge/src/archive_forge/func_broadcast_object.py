import os
import io
import itertools
from typing import (
import torch.distributed as dist
from .api import (
import torch
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._tensor import DTensor
from .metadata import (
def broadcast_object(self, object: Optional[T]) -> T:
    """Implement functionality similar to c10d::broadcast_object_list but without distributed enabled."""
    object_list = [object]
    if self.use_dist:
        dist.broadcast_object_list(object_list=object_list, group=self.group, src=self.coordinator_rank)
    return cast(T, object_list[0])