from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_op_strategy([aten.new_empty.default, aten.new_full.default, aten.new_ones.default, aten.new_zeros.default, aten.new_empty_strided.default], schema_info=RuntimeSchemaInfo(1, ['dtype']))
def new_factory_strategy(mesh: DeviceMesh, _) -> StrategyType:
    replica_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
    return OpStrategy([PlacementStrategy(replica_spec)])