from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def unshard_tensor_dim(placements: Sequence[Placement], dim: int) -> Tuple[Placement, ...]:
    """Disallow the given tensor dimension to be sharded."""
    return tuple((p if not isinstance(p, Shard) or p.dim != dim else Replicate() for p in placements))