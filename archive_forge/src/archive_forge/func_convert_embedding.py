import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('Embedding')
def convert_embedding(node, **kwargs):
    """Map MXNet's Embedding operator attributes to onnx's
    Gather operator."""
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get('axis', 0))
    dtype = str(attrs.get('dtype', 'float32'))
    nodes = [make_node('Cast', [input_nodes[0]], [name + '_indices_casted'], to=int(TensorProto.INT64)), make_node('Gather', [input_nodes[1], name + '_indices_casted'], [name], axis=axis, name=name)]
    return (nodes, (dtype,))