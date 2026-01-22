from dataclasses import dataclass
from typing import Callable, cast, Dict, Iterable, Optional, Sequence, Set, Tuple, Union
import torch
from torch import Tensor
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.api import Shard
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import DTensorSpec, Placement, Replicate
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
@register_prop_rule(aten_op_overload, schema_info=schema_info)
def reshape_prop(op_schema: OpSchema) -> OutputSharding:
    rules = spec.dim_map(*op_schema.args_schema, **op_schema.kwargs_schema)
    input_dtensor_spec = cast(DTensorSpec, op_schema.args_schema[0])
    mesh = input_dtensor_spec.mesh
    assert isinstance(input_dtensor_spec, DTensorSpec), 'Expected first input to be a DTensorSpec'
    global_in_shape = input_dtensor_spec.shape
    assert global_in_shape is not None, 'Shape required.'
    with disable_proxy_modes_tracing(), unset_fake_temporarily():
        global_out_shape, shard_out, shardable_dims = propagate_shape_and_sharding(input_dtensor_spec.placements, tuple(global_in_shape), rules, mesh.shape)
    if shard_out is not None:
        output_dtensor_spec = DTensorSpec(mesh=mesh, placements=tuple(shard_out))
        args = op_schema.args_schema
        shape_argnum = spec.shape_argnum
        if shape_argnum is not None:
            local_out_shape = compute_local_shape(list(global_out_shape), mesh, shard_out)
            suggested_schema = OpSchema(op=op_schema.op, args_schema=args[:shape_argnum] + (tuple(local_out_shape),) + args[shape_argnum + 1:], kwargs_schema=op_schema.kwargs_schema)
            return OutputSharding(output_spec=output_dtensor_spec, schema_suggestions=[suggested_schema], needs_redistribute=True)
        return OutputSharding(output_spec=output_dtensor_spec)
    else:
        suggested_placements = [p if not isinstance(p, Shard) or shardable_dims[p.dim][mesh_dim] else Replicate() for mesh_dim, p in enumerate(input_dtensor_spec.placements)]
        return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(DTensorSpec(placements=tuple(suggested_placements), mesh=input_dtensor_spec.mesh, tensor_meta=input_dtensor_spec.tensor_meta),) + op_schema.args_schema[1:], kwargs_schema=op_schema.kwargs_schema)])