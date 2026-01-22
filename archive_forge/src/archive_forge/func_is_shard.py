from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.distributed.device_mesh import DeviceMesh
def is_shard(self, dim: Optional[int]=None) -> bool:
    is_shard_instance = isinstance(self, Shard)
    if dim is not None and is_shard_instance:
        return cast(Shard, self).dim == dim
    else:
        return is_shard_instance