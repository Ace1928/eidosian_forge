import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('tile')
def convert_tile(node, **kwargs):
    """Map MXNet's Tile operator attributes to onnx's Tile
    operator and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    data = input_nodes[0]
    reps = convert_string_to_list(attrs['reps'])
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([1], name + '_1', kwargs['initializer'])
    create_tensor(reps, name + '_reps', kwargs['initializer'], dtype='int64')
    create_tensor([len(reps)], name + '_reps_len', kwargs['initializer'])
    nodes = [make_node('Shape', [data], [name + '_data_shape']), make_node('Shape', [name + '_data_shape'], [name + '_data_dim']), make_node('Max', [name + '_data_dim', name + '_reps_len'], [name + '_max']), make_node('Sub', [name + '_max', name + '_data_dim'], [name + '_data_diff']), make_node('Concat', [name + '_data_diff', name + '_0'], [name + '_concat0_out'], axis=0), make_node('Pad', [name + '_data_shape', name + '_concat0_out', name + '_1'], [name + '_data_shape_pad']), make_node('Reshape', [data, name + '_data_shape_pad'], [name + '_data']), make_node('Sub', [name + '_max', name + '_reps_len'], [name + '_reps_diff']), make_node('Concat', [name + '_reps_diff', name + '_0'], [name + '_concat1_out'], axis=0), make_node('Pad', [name + '_reps', name + '_concat1_out', name + '_1'], [name + '_reps_pad']), make_node('Tile', [name + '_data', name + '_reps_pad'], [name], name=name)]
    return nodes