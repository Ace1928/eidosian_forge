import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('broadcast_greater_equal')
def convert_broadcast_greater_equal(node, **kwargs):
    """Map MXNet's broadcast_greater_equal operator
    """
    from onnx.helper import make_node
    name, input_nodes, _ = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    nodes = [make_node('GreaterOrEqual', [input_nodes[0], input_nodes[1]], [name + '_gt']), make_node('Cast', [name + '_gt'], [name], to=dtype_t)]
    return nodes