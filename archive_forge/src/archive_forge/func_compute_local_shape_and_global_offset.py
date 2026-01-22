from typing import cast, List, Sequence, Tuple
import torch
from torch._prims_common import ShapeType
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def compute_local_shape_and_global_offset(global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute the local tensor shape and the global offsets into the original tensor
    of a DTensor on its current global rank. This is useful for checkpointing purpose.

    Example (2 host with 4GPUs each):
    # Below is a DeviceMesh with mesh_shape of (2, 4)
    mesh = DeviceMesh(device_type="cuda",
                        mesh=[
                        [0, 1, 2, 3],
                        [4, 5, 6, 7]
                        ],
    )

    Let's say we distribute a global_tensor of shape (8,4) over the above DeviceMesh
    with a placements of [Shard(0), Shard(0)].
    The local shape and global offset will be as follows:
    rank0 -- local_shape:[1, 4], global_offset:[0, 0]
    rank1 -- local_shape:[1, 4], global_offset:[1, 0]
    rank2 -- local_shape:[1, 4], global_offset:[2, 0]
    rank5 -- local_shape:[1, 4], global_offset:[5, 0]
    rank3 -- local_shape:[1, 4], global_offset:[3, 0]
    rank4 -- local_shape:[1, 4], global_offset:[4, 0]
    rank6 -- local_shape:[1, 4], global_offset:[6, 0]
    rank7 -- local_shape:[1, 4], global_offset:[7, 0]
    """
    my_coordinate = mesh.get_coordinate()
    if my_coordinate is None:
        return ((), ())
    else:
        local_shape = list(global_shape)
        global_offset = [0] * len(global_shape)
        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                local_offset = [0] * len(global_shape)
                assert shard_dim < len(local_shape), f'Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}'
                shard_size, shard_offset = placement._local_shard_size_on_dim(local_shape[shard_dim], mesh_dim_size, my_coordinate[idx], return_offset=True)
                local_shape[shard_dim] = shard_size
                local_offset[shard_dim] = shard_offset
                if global_offset[shard_dim] <= local_offset[shard_dim]:
                    global_offset[shard_dim] = local_offset[shard_dim]
                else:
                    global_offset[shard_dim] += local_offset[shard_dim]
        return (tuple(local_shape), tuple(global_offset))