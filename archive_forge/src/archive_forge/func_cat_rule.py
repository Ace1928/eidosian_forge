from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_prop_rule(aten.cat.default, schema_info=RuntimeSchemaInfo(1, needs_pytree=True))
def cat_rule(op_schema: OpSchema) -> OutputSharding:

    def is_empty(spec: DTensorSpec) -> bool:
        return list(spec.shape) == [0]
    tensor_list_specs = cast(List[DTensorSpec], op_schema.args_schema[0])
    assert len(tensor_list_specs) > 0, 'torch.cat expects a non-empty list of tensors'
    non_empty_specs = [spec for spec in tensor_list_specs if not is_empty(spec)]
    if len(non_empty_specs) == 0:
        return OutputSharding(output_spec=DTensorSpec(mesh=tensor_list_specs[0].mesh, placements=tensor_list_specs[0].placements))
    assert all((spec.ndim == non_empty_specs[0].ndim for spec in non_empty_specs)), f'Expect all tensors to have same shape or empty, but got {tensor_list_specs}'
    assert all((spec.mesh == tensor_list_specs[0].mesh for spec in tensor_list_specs)), f'Expect all tensors to have same mesh, but got {tensor_list_specs}'
    ndim = 1
    for spec in tensor_list_specs:
        ndim = max(ndim, spec.ndim)
    dim = 0
    if len(op_schema.args_schema) > 1:
        dim = cast(int, op_schema.args_schema[1])
    dim = normalize_dim(dim, ndim)
    need_reshard = False
    tensor_list_specs_after: List[DTensorSpec] = []
    for spec in tensor_list_specs:
        if not is_empty(spec) and (is_tensor_dim_sharded(spec, dim=dim) or is_tensor_partial(spec)):
            need_reshard = True
            tensor_list_specs_after.append(DTensorSpec(mesh=spec.mesh, placements=replicate_tensor_dim(spec.placements, dim=dim), tensor_meta=spec.tensor_meta))
        else:
            tensor_list_specs_after.append(spec)
    tensor_list_specs = tensor_list_specs_after
    non_empty_specs = [spec for spec in tensor_list_specs if not is_empty(spec)]
    mesh = non_empty_specs[0].mesh
    ndim = non_empty_specs[0].ndim
    new_placements: List[Placement] = []
    for mesh_dim in range(mesh.ndim):
        if any((spec.placements[mesh_dim] != non_empty_specs[0].placements[mesh_dim] for spec in non_empty_specs)):
            need_reshard = True
            reshard_cost = []
            for shard_dim in range(ndim):
                cost: float = 0.0
                for spec in non_empty_specs:
                    global_shape = spec.shape
                    if global_shape[shard_dim] < mesh.size(mesh_dim):
                        cost = +float('inf')
                    elif is_tensor_dim_sharded(spec, dim=shard_dim) or prod(global_shape) == 0:
                        continue
                    else:
                        local_shape = compute_local_shape(global_shape, spec.mesh, spec.placements)
                        cost += prod(local_shape) * spec.mesh.size(mesh_dim)
                reshard_cost.append(cost)
            best_dim = reshard_cost.index(min(reshard_cost))
            new_placements.append(Shard(best_dim))
        else:
            new_placements.append(non_empty_specs[0].placements[mesh_dim])
    if need_reshard:
        tensor_list_specs_after = []
        for spec in tensor_list_specs:
            if is_empty(spec):
                tensor_list_specs_after.append(spec)
            else:
                tensor_list_specs_after.append(DTensorSpec(mesh=spec.mesh, placements=tuple(new_placements), tensor_meta=spec.tensor_meta))
        return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(tuple(tensor_list_specs_after), *op_schema.args_schema[1:]), kwargs_schema=op_schema.kwargs_schema)])
    else:
        return OutputSharding(output_spec=DTensorSpec(mesh=non_empty_specs[0].mesh, placements=non_empty_specs[0].placements))