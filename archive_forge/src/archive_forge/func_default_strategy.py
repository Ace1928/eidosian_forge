from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def default_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    select_strategy = op_schema.args_schema[0]
    assert isinstance(select_strategy, OpStrategy)
    default_strategy = []
    for strategy in select_strategy.strategies:
        default_strategy.append(PlacementStrategy(output_spec=DTensorSpec(mesh=strategy.output_spec.mesh, placements=strategy.output_spec.placements)))
    return OpStrategy(default_strategy)