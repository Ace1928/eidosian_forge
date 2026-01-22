from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_prop_rule([aten.split.Tensor, aten.split_with_sizes.default], schema_info=RuntimeSchemaInfo(1))
def split_rule(op_schema: OpSchema) -> OutputSharding:
    output_spec_list: List[DTensorSpec] = []
    input_spec = cast(DTensorSpec, op_schema.args_schema[0])
    ndim = input_spec.ndim
    split_size_or_sections = op_schema.args_schema[1]
    dim = cast(int, op_schema.args_schema[2]) if len(op_schema.args_schema) > 2 else 0
    dim = normalize_dim(dim, ndim)
    if input_spec.sums:
        raise NotImplementedError(f'splitting distributed tensor with _Partial placement is not implemented!\nDTensorSpec={input_spec}')
    need_reshard = False
    if is_tensor_dim_sharded(input_spec, dim=dim):
        need_reshard = True
        input_spec = DTensorSpec(mesh=input_spec.mesh, placements=unshard_tensor_dim(input_spec.placements, dim=dim), tensor_meta=input_spec.tensor_meta)
    if need_reshard:
        return OutputSharding(None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(input_spec,) + op_schema.args_schema[1:], kwargs_schema=op_schema.kwargs_schema)])

    def size_split(N, i):
        assert i > 0
        return [i] * (N // i) + ([N % i] if N % i != 0 else [])
    output_size_list = size_split(input_spec.shape[dim], split_size_or_sections) if isinstance(split_size_or_sections, int) else split_size_or_sections
    output_spec_list = [DTensorSpec(mesh=input_spec.mesh, placements=input_spec.placements) for _ in range(len(output_size_list))]
    return OutputSharding(output_spec_list)