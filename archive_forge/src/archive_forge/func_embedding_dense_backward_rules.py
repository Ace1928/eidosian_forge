import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule(aten.embedding_dense_backward.default)
def embedding_dense_backward_rules(op_schema: OpSchema) -> OutputSharding:
    grad_output, indices = op_schema.args_schema[:2]
    assert isinstance(grad_output, DTensorSpec)
    assert isinstance(indices, DTensorSpec)
    if grad_output.placements == indices.placements:
        return OutputSharding(output_spec=DTensorSpec(mesh=indices.mesh, placements=(_Partial(),)))
    elif grad_output.placements == [_Partial()] and indices.placements == [Replicate()]:
        return OutputSharding(output_spec=DTensorSpec(mesh=indices.mesh, placements=(_Partial(),)))
    elif all((placement.is_replicate() for placement in indices.placements)):
        return OutputSharding(output_spec=DTensorSpec(mesh=indices.mesh, placements=(Shard(1),)))
    else:
        raise NotImplementedError(f'Unsupported embedding dense backward schema:\ngrad_output - {grad_output}\nindices - {indices}')