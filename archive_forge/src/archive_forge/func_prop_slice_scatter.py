from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule(aten.slice_scatter.default)
def prop_slice_scatter(op_schema: OpSchema) -> OutputSharding:
    defaults = (None, None, 0, None, None, 1)
    input, src, dim, start, end, step = op_schema.args_schema + defaults[len(op_schema.args_schema):]
    assert isinstance(input, DTensorSpec)
    assert isinstance(src, DTensorSpec)
    assert isinstance(dim, int)
    if dim < 0:
        dim += input.ndim
    if input.shape[dim] == src.shape[dim]:
        assert start == 0
        assert end >= src.shape[dim]
        dim = None
    input_suggestion = list(_refine_sharding(op_schema, dim))
    for i, p in enumerate(input_suggestion):
        if isinstance(p, Shard) and p.dim == dim:
            input_suggestion[i] = Replicate()
    input_suggestion = tuple(input_suggestion)
    if input_suggestion == tuple(input.placements) and src.placements == tuple(input.placements):
        return OutputSharding(output_spec=DTensorSpec(mesh=input.mesh, placements=input.placements))
    else:
        return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(DTensorSpec(mesh=input.mesh, placements=input_suggestion, tensor_meta=input.tensor_meta), DTensorSpec(mesh=src.mesh, placements=input_suggestion, tensor_meta=src.tensor_meta)) + op_schema.args_schema[2:], kwargs_schema=op_schema.kwargs_schema)])