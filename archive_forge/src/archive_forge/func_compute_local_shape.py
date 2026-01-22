from typing import cast, List, Sequence, Tuple
import torch
from torch._prims_common import ShapeType
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def compute_local_shape(global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]) -> Tuple[int, ...]:
    """
    Compute the shape of a local shard of the given DTensor on its current
    coordinate of the mesh.
    """
    my_coordinate = mesh.get_coordinate()
    if my_coordinate is None:
        return (0,)
    else:
        local_shape = list(global_shape)
        ndim = len(global_shape)
        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                assert shard_dim < ndim, f'Sharding dim {shard_dim} greater than tensor ndim {ndim}'
                local_shard_size, _ = placement._local_shard_size_on_dim(local_shape[shard_dim], mesh_dim_size, my_coordinate[idx])
                assert isinstance(local_shard_size, int)
                local_shape[shard_dim] = local_shard_size
        return tuple(local_shape)