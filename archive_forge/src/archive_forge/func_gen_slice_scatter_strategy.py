from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_op_strategy(aten.slice_scatter.default, schema_info=RuntimeSchemaInfo(2))
def gen_slice_scatter_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    input_strategy = op_schema.args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    input_ndim = input_strategy.output_ndim
    slice_dim = cast(int, op_schema.args_schema[2]) if len(op_schema.args_schema) > 2 else 0
    slice_dim = normalize_dim(slice_dim, input_ndim)
    slice_scatter_strategy = OpStrategy([])
    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if not (is_tensor_dim_sharded(arg_spec, dim=slice_dim) or is_tensor_partial(arg_spec)):
            slice_scatter_strategy.strategies.append(PlacementStrategy(output_spec=arg_spec))
    if not slice_scatter_strategy.strategies:
        for arg_strategy in input_strategy.strategies:
            arg_spec = arg_strategy.output_spec
            replicate_spec = DTensorSpec(mesh, replicate_tensor_dim(arg_spec.placements, dim=slice_dim))
            slice_scatter_strategy.strategies.append(PlacementStrategy(output_spec=replicate_spec))
    return slice_scatter_strategy