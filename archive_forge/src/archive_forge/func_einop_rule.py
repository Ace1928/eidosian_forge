from typing import cast, Dict, List, Optional, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import prod
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
def einop_rule(equation: str, op_schema: OpSchema, *, linearity: bool=False, enforce_sharding: Optional[Dict[str, int]]=None) -> OutputSharding:
    """
    Propagate the sharding of inputs to output for ops whose data moves according to einsum notation.

    This is mostly borrowed from @zdevito's sharding simulator. Examples:
        mk,kn->mn - einsum
        ij,ij->ij - addition
        ij,j->ij - broadcasted addition
        ij->i - reduction
    Other ops could use this propagation algorithm when applied, note
    that einsum propagation only deal with list of specs (DTensor specs)
    as it only works on list of tensors!

    linearity in einop_rule means that the calling op `f` follows this rule:
        f(a + b) = f(a) + f(b)

    In this case we can propagate the partial sum, note that linearity in einop
    only applies to partial sum, not other operations like min/max (which are
    associative but not linear).
    """
    inputs, outputs = equation.split('->')
    input_dims, output_dims = (inputs.split(','), outputs.split(','))
    input_specs = op_schema.args_spec
    output_dim = output_dims[0]
    dim_to_sharding: Dict[str, int] = {}
    dim_to_size: Dict[str, int] = {}
    pending_sums_counter: Dict[int, int] = {}
    seen_shardings: Dict[int, str] = {}
    needs_reshard = False

    def merge_sharding(dim: str, a: int, b: int) -> int:
        if a != b:
            if a == -1 or b == -1:
                nonlocal needs_reshard
                needs_reshard = True
                return a if a != -1 else b
            else:
                raise RuntimeError(f'{equation}: dim {dim} sharded two different ways: {a} and {b}')
        else:
            return a
    for input_dim, input_spec in zip(input_dims, input_specs):
        input_sums = input_spec.sums
        for sum_dim in input_sums:
            if sum_dim not in pending_sums_counter:
                seen_shardings[sum_dim] = '+'
            pending_sums_counter[sum_dim] = pending_sums_counter.get(sum_dim, 0) + 1
        for idx, (dim, mesh_dim) in enumerate(zip(input_dim, input_spec.dim_map)):
            if enforce_sharding and dim in enforce_sharding:
                if enforce_sharding[dim] != mesh_dim:
                    needs_reshard = True
                dim_to_sharding[dim] = enforce_sharding[dim]
                dim_to_size[dim] = input_spec.shape[idx]
            elif dim not in dim_to_sharding:
                dim_to_sharding[dim] = mesh_dim
                dim_to_size[dim] = input_spec.shape[idx]
            else:
                dim_to_sharding[dim] = merge_sharding(dim, dim_to_sharding[dim], mesh_dim)
                assert dim_to_size[dim] == input_spec.shape[idx]
            merged_sharding_for_dim = dim_to_sharding[dim]
            if merged_sharding_for_dim != -1:
                if merged_sharding_for_dim in seen_shardings and dim != seen_shardings[merged_sharding_for_dim]:
                    needs_reshard = True
                    seen_shardings[merged_sharding_for_dim] += dim
                else:
                    seen_shardings[merged_sharding_for_dim] = dim
    if pending_sums_counter and (not linearity):
        return _gen_reshard_suggestions(op_schema, input_dims, input_specs, dim_to_sharding, [])
    else:
        for value in pending_sums_counter.values():
            if value != len(input_specs):
                needs_reshard = True
    for mesh_dim, dims in seen_shardings.items():
        if len(dims) > 1:
            costs = []
            for d in dims:
                cost = 0
                for input_dim, input_spec in zip(input_dims, input_specs):
                    if d in input_dim and input_spec.dim_map[input_dim.index(d)] == mesh_dim:
                        assert input_spec.tensor_meta is not None
                        global_shape = input_spec.tensor_meta.shape
                        local_shape = compute_local_shape(global_shape, input_spec.mesh, input_spec.placements)
                        cost += prod(local_shape) * input_spec.mesh.size(mesh_dim)
                costs.append(cost)
            d_to_keep_sharding = dims[costs.index(max(costs))]
            for d in dims:
                if d != d_to_keep_sharding:
                    dim_to_sharding[d] = -1
    pending_sums = list(pending_sums_counter.keys())
    if needs_reshard:
        return _gen_reshard_suggestions(op_schema, input_dims, input_specs, dim_to_sharding, pending_sums)
    for dim, shard_on_mesh in dim_to_sharding.items():
        if dim not in output_dims[0] and shard_on_mesh != -1:
            pending_sums.append(shard_on_mesh)
    output_dim_map = []
    output_shape = []
    for dim in output_dim:
        if dim == '1':
            output_dim_map.append(-1)
            output_shape.append(1)
        else:
            output_dim_map.append(dim_to_sharding[dim])
            output_shape.append(dim_to_size[dim])
    assert input_specs[0].tensor_meta is not None
    tensor_meta = TensorMeta(torch.Size(output_shape), input_specs[0].tensor_meta.stride, input_specs[0].tensor_meta.dtype)
    return OutputSharding(DTensorSpec.from_dim_map(input_specs[0].mesh, output_dim_map, pending_sums, tensor_meta=tensor_meta))