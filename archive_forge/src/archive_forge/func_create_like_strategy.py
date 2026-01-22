from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_op_strategy([aten.empty_like.default, aten.ones_like.default, aten.rand_like.default, aten.randn_like.default, aten.zeros_like.default], schema_info=RuntimeSchemaInfo(1, ['dtype']))
@register_op_strategy([aten.full_like.default], schema_info=RuntimeSchemaInfo(2, ['dtype']))
@register_op_strategy([aten.randint_like.default, aten.randint_like.low_dtype, aten.randint_like.low_dtype_out], schema_info=RuntimeSchemaInfo(3, ['dtype']))
def create_like_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    select_strategy = op_schema.args_schema[0]
    create_like_strategy = OpStrategy([])
    assert isinstance(select_strategy, OpStrategy)
    for arg_strategy in select_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_partial(arg_spec):
            output_spec = DTensorSpec(mesh=arg_spec.mesh, placements=tuple((Replicate() if isinstance(p, _Partial) else p for p in arg_spec.placements)))
            create_like_strategy.strategies.append(PlacementStrategy(output_spec=output_spec, input_specs=(arg_spec,)))
        else:
            create_like_strategy.strategies.append(PlacementStrategy(arg_spec))
    return create_like_strategy