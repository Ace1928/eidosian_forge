from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
@register_prop_rule(aten.index_select.default, schema_info=RuntimeSchemaInfo(1))
def prop_index_select(op_schema: OpSchema) -> OutputSharding:
    values_spec, dim, indices_spec = op_schema.args_schema
    assert isinstance(values_spec, DTensorSpec)
    assert isinstance(dim, int)
    assert isinstance(indices_spec, DTensorSpec)
    all_indices_spec: List[Optional[DTensorSpec]] = [indices_spec if dim == i else None for i in range(values_spec.ndim)]
    result = prop_index(OpSchema(op=op_schema.op, args_schema=(values_spec, all_indices_spec), kwargs_schema=op_schema.kwargs_schema))
    if result.schema_suggestions:
        result.schema_suggestions = [OpSchema(op=op_schema.op, args_schema=(s.args_schema[0], dim, s.args_schema[1][dim]), kwargs_schema=op_schema.kwargs_schema) for s in result.schema_suggestions]
    return result