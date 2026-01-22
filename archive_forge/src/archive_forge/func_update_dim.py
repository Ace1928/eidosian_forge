from typing import Any, Dict, List, Set
import onnx.checker
from onnx import ModelProto, ValueInfoProto
def update_dim(tensor: ValueInfoProto, dim: Any, j: int, name: str) -> None:
    dim_proto = tensor.type.tensor_type.shape.dim[j]
    if isinstance(dim, int):
        if dim >= 0:
            if dim_proto.HasField('dim_value') and dim_proto.dim_value != dim:
                raise ValueError(f'Unable to set dimension value to {dim} for axis {j} of {name}. Contradicts existing dimension value {dim_proto.dim_value}.')
            dim_proto.dim_value = dim
        else:
            generated_dim_param = name + '_' + str(j)
            if generated_dim_param in dim_param_set:
                raise ValueError(f'Unable to generate unique dim_param for axis {j} of {name}. Please manually provide a dim_param value.')
            dim_proto.dim_param = generated_dim_param
    elif isinstance(dim, str):
        dim_proto.dim_param = dim
    else:
        raise ValueError(f'Only int or str is accepted as dimension value, incorrect type: {type(dim)}')