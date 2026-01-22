import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('log2')
def convert_log2(node, **kwargs):
    """Map MXNet's log2 operator attributes to onnx's operator.
    """
    from onnx.helper import make_node, make_tensor
    name, input_nodes, _ = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    ln2 = np.array([0.6931471805599453], dtype=dtype)
    if dtype == 'float16':
        ln2 = ln2.view(dtype=np.uint16)
    ln2v = make_tensor(name + '_ln2', dtype_t, [1], ln2)
    nodes = [make_node('Log', [input_nodes[0]], [name + '_log']), make_node('Constant', [], [name + '_ln2'], value=ln2v), make_node('Div', [name + '_log', name + '_ln2'], [name], name=name)]
    return nodes