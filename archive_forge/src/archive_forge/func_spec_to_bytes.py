import logging
import math
from typing import List, Optional
import torch
import torch.distributed._tensor.placement_types as placement_types
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import (
def spec_to_bytes(spec: 'placement_types.DTensorSpec') -> int:
    assert spec.tensor_meta is not None, 'spec should have tensor meta defined!'
    return spec.tensor_meta.dtype.itemsize * math.prod(spec.shape)