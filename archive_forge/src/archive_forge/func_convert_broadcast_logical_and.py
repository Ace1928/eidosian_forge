import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('broadcast_logical_and')
def convert_broadcast_logical_and(node, **kwargs):
    """Map MXNet's broadcast logical and operator attributes to onnx's And operator
    and return the created node.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, _ = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    nodes = [make_node('Cast', [input_nodes[0]], [name + '_cast0'], to=int(TensorProto.BOOL)), make_node('Cast', [input_nodes[1]], [name + '_cast1'], to=int(TensorProto.BOOL)), make_node('And', [name + '_cast0', name + '_cast1'], [name + '_and']), make_node('Cast', [name + '_and'], [name], name=name, to=int(dtype_t))]
    return nodes