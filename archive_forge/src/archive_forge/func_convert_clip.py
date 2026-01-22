import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('clip')
def convert_clip(node, **kwargs):
    """Map MXNet's Clip operator attributes to onnx's Clip operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    opset_version = kwargs['opset_version']
    a_min = float(attrs.get('a_min', -np.inf))
    a_max = float(attrs.get('a_max', np.inf))
    if opset_version >= 11:
        input_dtype = get_input_dtypes(node, kwargs)[0]
        create_const_scalar_node(name + '_min', np.float32(a_min).astype(input_dtype), kwargs)
        create_const_scalar_node(name + '_max', np.float32(a_max).astype(input_dtype), kwargs)
        nodes = [make_node('Clip', [input_nodes[0], name + '_min', name + '_max'], [name], name=name)]
    else:
        nodes = [make_node('Clip', input_nodes, [name], name=name, min=a_min, max=a_max)]
    return nodes