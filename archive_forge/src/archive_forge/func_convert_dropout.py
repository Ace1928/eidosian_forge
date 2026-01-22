import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('Dropout')
def convert_dropout(node, **kwargs):
    """Map MXNet's Dropout operator attributes to onnx's Dropout operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    _ = float(attrs.get('p', 0.5))
    _ = convert_string_to_list(attrs.get('axes', 'None'))
    mode = attrs.get('mode', 'training')
    if mode != 'training':
        raise NotImplementedError("Dropout does not currently support mode!='training'")
    nodes = [make_node('Identity', [input_nodes[0]], [name])]
    return nodes