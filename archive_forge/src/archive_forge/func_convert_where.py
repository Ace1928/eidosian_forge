import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('where')
def convert_where(node, **kwargs):
    """Map MXNet's where operator attributes to onnx's Where
    operator and return the created node.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, _ = get_inputs(node, kwargs)
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([1], name + '_1', kwargs['initializer'])
    nodes = [make_node('Shape', [input_nodes[0]], [name + '_cond_shape']), make_node('Shape', [name + '_cond_shape'], [name + '_cond_dim']), make_node('Shape', [input_nodes[1]], [name + '_x_shape']), make_node('Shape', [name + '_x_shape'], [name + '_x_dim']), make_node('Sub', [name + '_x_dim', name + '_cond_dim'], [name + '_sub']), make_node('Concat', [name + '_0', name + '_sub'], [name + '_concat'], axis=0), make_node('Pad', [name + '_cond_shape', name + '_concat', name + '_1'], [name + '_cond_new_shape']), make_node('Reshape', [input_nodes[0], name + '_cond_new_shape'], [name + '_cond']), make_node('Cast', [name + '_cond'], [name + '_bool'], to=int(TensorProto.BOOL)), make_node('Where', [name + '_bool', input_nodes[1], input_nodes[2]], [name], name=name)]
    return nodes