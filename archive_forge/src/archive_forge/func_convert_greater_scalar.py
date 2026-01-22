import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_greater_scalar')
def convert_greater_scalar(node, **kwargs):
    """Map MXNet's greater_scalar operator attributes to onnx's Greater
    operator and return the created node.
    """
    from onnx.helper import make_node, make_tensor
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    scalar = float(attrs.get('scalar'))
    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    if str(dtype).startswith('int'):
        scalar = int(scalar)
    elif dtype == 'float16':
        scalar = np.float16(scalar).view(np.uint16)
    tensor_value = make_tensor(name + '_scalar', dtype_t, [1], [scalar])
    nodes = [make_node('Constant', [], [name + '_rhs'], value=tensor_value), make_node('Greater', [input_nodes[0], name + '_rhs'], [name + '_gt']), make_node('Cast', [name + '_gt'], [name], to=dtype_t, name=name)]
    return nodes