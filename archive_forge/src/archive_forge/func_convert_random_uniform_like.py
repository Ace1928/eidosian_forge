import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_random_uniform_like')
def convert_random_uniform_like(node, **kwargs):
    """Map MXNet's random_uniform_like operator attributes to onnx's RandomUniformLike operator
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    low = float(attrs.get('low', 0.0))
    high = float(attrs.get('high', 1.0))
    nodes = [make_node('RandomUniformLike', [input_nodes[0]], [name], name=name, dtype=dtype_t, low=low, high=high)]
    return nodes