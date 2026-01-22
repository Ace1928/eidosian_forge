import warnings
from typing import Callable, cast, Optional, Sequence, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed._tensor.dispatch as op_dispatch
import torch.distributed._tensor.random as random
import torch.nn as nn
from torch.distributed._tensor._collective_utils import mesh_broadcast
from torch.distributed._tensor._utils import compute_global_tensor_info
from torch.distributed._tensor.placement_types import (
from torch.distributed._tensor.random import (
from torch.distributed._tensor.redistribute import (
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
@staticmethod
def from_local(local_tensor: torch.Tensor, device_mesh: Optional[DeviceMesh]=None, placements: Optional[Sequence[Placement]]=None, *, run_check: bool=True, shape: Optional[torch.Size]=None, stride: Optional[Tuple[int, ...]]=None) -> 'DTensor':
    """
        Create a :class:`DTensor` from a local torch.Tensor on each rank
        according to the `device_mesh` and `placements` specified.

        Args:
            local_tensor (torch.Tensor): local torch.Tensor on each rank.
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                tensor, if not specified, must be called under a DeviceMesh
                context manager, default: None
            placements (List[:class:`Placement`], optional): the placements that
                describes how to place the local torch.Tensor on DeviceMesh, must
                have the same number of elements as `device_mesh.ndim`. If not
                specified, we will by default replicate the tensor across the
                `device_mesh` from the first rank of each dimension of the `device_mesh`.

        Keyword args:
            run_check (bool, optional): indicate whether to run check across ranks
                to check meta information and data. if have :class:`Replicate` in
                `placements`, the data on first rank of the device mesh dimension
                will be broadcasted to other ranks.
            shape (torch.Size, optional): A List of int which specifies the size of
                DTensor which build on top of `local_tensor`. Note this needs to be
                provided if the shape of `local_tensor` are different across the ranks.
                If not provided, `shape` will be computed assuming the given distributed
                tensor is evenly sharded across ranks.
            stride (tuple, optional): A List of int which specifies the stride of DTensor.
                If not provided, `stride` will be computed assuming the given distributed
                tensor is evenly sharded across ranks.

        Returns:
            A :class:`DTensor` object

        .. note:: `from_local` is differentiable, the `requires_grad` of the created
            `DTensor` object will depend on if `local_tensor` requires_grad or not.
        """
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    device_type = device_mesh.device_type
    if device_type != local_tensor.device.type and (not local_tensor.is_meta):
        local_tensor = local_tensor.to(device_type)
    if placements is None:
        placements = [Replicate() for _ in range(device_mesh.ndim)]
    else:
        placements = list(placements)
        for idx, placement in enumerate(placements):
            if placement.is_shard():
                placement = cast(Shard, placement)
                if placement.dim < 0:
                    placements[idx] = Shard(placement.dim + local_tensor.ndim)
    return _FromTorchTensor.apply(local_tensor, device_mesh, tuple(placements), run_check, shape, stride)