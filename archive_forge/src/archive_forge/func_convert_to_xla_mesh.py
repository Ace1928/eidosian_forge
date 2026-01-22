import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Placement, Replicate
from torch.distributed.device_mesh import DeviceMesh
@with_xla
def convert_to_xla_mesh(dt_mesh: DeviceMesh) -> 'Mesh':
    """
    Convert DTensor `dt_mesh` to XLAShardedTensor `partition_spec`.

    Example (1x4 logical device mesh topology):
      ```
      dt_mesh = DeviceMesh("xla", [[1, 2, 3, 4]])
      dt_mesh.shape
      >> torch.Size([1, 4])

      mesh = convert_to_xla_mesh(dt_mesh)
      mesh_shape
      >> [1, 4]
      ```
    """
    assert dt_mesh.size() == xr.global_runtime_device_count()
    return Mesh(dt_mesh.mesh.flatten(), tuple(dt_mesh.mesh.size()), dt_mesh.mesh_dim_names)