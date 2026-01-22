from typing import cast, List, Optional, Sequence, Tuple
import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def common_reduction_strategy(mesh: DeviceMesh, input_strategy: OpStrategy, reduce_dims: List[int], keep_dim: bool=False, reduction_linear: bool=True, reduction_op: c10d.ReduceOp.RedOpType=c10d.ReduceOp.SUM) -> OpStrategy:
    """
    reduction_linear means that the reduction `f` follows this rule:
        f([f(a), f(b)]) = f([a, b])

    reduction linear should be super set of linearity.
    """
    reduction_strategy = OpStrategy([])
    for strtg in input_strategy.strategies:
        if not reduction_linear:
            input_placements = replicate_reduction_dims(strtg.output_spec.placements, reduce_dims)
        else:
            input_placements = strtg.output_spec.placements
        input_spec = DTensorSpec(mesh=mesh, placements=input_placements, tensor_meta=strtg.output_spec.tensor_meta)
        reduce_dims_map = _infer_reduce_dims_map(reduce_dims, input_spec.ndim, keep_dim)
        out_placements = map_placements_after_reduction(input_spec.placements, reduce_dims, reduce_dims_map, reduction_op)
        redistribute_cost = [generate_redistribute_costs(input_strategy, input_spec)]
        reduction_strategy.strategies.append(PlacementStrategy(output_spec=DTensorSpec(mesh=mesh, placements=out_placements), input_specs=(input_spec,), redistribute_cost=redistribute_cost))
    return reduction_strategy