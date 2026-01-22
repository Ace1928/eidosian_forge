import logging
import math
from typing import List, Optional
import torch
import torch.distributed._tensor.placement_types as placement_types
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import (
def reduce_scatter_cost(num_bytes: float, mesh: DeviceMesh, mesh_dim: int) -> float:
    num_devices_on_mesh_dim = mesh.size(mesh_dim)
    bandwidth_factor = get_bandwidth_factor(mesh)[mesh_dim]
    return 1 + bandwidth_factor * num_bytes * (num_devices_on_mesh_dim - 1) / num_devices_on_mesh_dim