import logging
import math
from typing import List, Optional
import torch
import torch.distributed._tensor.placement_types as placement_types
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import (
def get_bandwidth_factor(mesh: DeviceMesh) -> List[float]:
    factors = [1.0] * mesh.ndim
    num_devices_per_host = _mesh_resources.num_devices_per_host(mesh.device_type)
    num_devices = 1
    for mesh_dim in reversed(range(mesh.ndim)):
        num_devices *= mesh.size(mesh_dim)
        if num_devices <= num_devices_per_host:
            factors[mesh_dim] = 0.2
    return factors