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
@property
def placements(self) -> Sequence[Placement]:
    """
        The placements attribute of this DTensor that describes the layout of this
        DTensor on the its DeviceMesh.

        .. note:: placements is a read-only property, it can not be set.
        """
    return self._spec.placements