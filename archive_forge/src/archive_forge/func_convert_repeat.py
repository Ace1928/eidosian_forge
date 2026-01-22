import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('repeat')
def convert_repeat(node, **kwargs):
    """Map MXNet's repeat operator attributes to onnx's Tile operator.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')
    repeats = int(attrs.get('repeats', 1))
    axis = attrs.get('axis', 'None')
    if repeats <= 0:
        raise NotImplementedError('repeat operator does not support parameter repeats==0')
    nodes = []
    if axis == 'None':
        create_tensor([repeats], name + '_rep', kwargs['initializer'])
        create_tensor([1, repeats], name + '_repeats', kwargs['initializer'])
        nodes += [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('ReduceProd', [name + '_shape'], [name + '_size']), make_node('Reshape', [input_nodes[0], name + '_size'], [name + '_flat']), make_node('Unsqueeze', [name + '_flat'], [name + '_unsqueeze'], axes=[-1]), make_node('Tile', [name + '_unsqueeze', name + '_repeats'], [name + '_tile']), make_node('Mul', [name + '_size', name + '_rep'], [name + '_new_size']), make_node('Reshape', [name + '_tile', name + '_new_size'], [name], name=name)]
    else:
        axis = int(axis)
        repeats -= 1
        create_tensor([repeats], name + '_repeats', kwargs['initializer'])
        create_tensor([1], name + '_1', kwargs['initializer'])
        create_tensor([0], name + '_0', kwargs['initializer'])
        create_tensor([axis], name + '_axis', kwargs['initializer'])
        create_const_scalar_node(name + '_0_s', np.int64(0), kwargs)
        create_const_scalar_node(name + '_1_s', np.int64(1), kwargs)
        nodes += [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Shape', [name + '_shape'], [name + '_dim']), make_node('Squeeze', [name + '_dim'], [name + '_dim_s'], axes=[0]), make_node('Range', [name + '_0_s', name + '_dim_s', name + '_1_s'], [name + '_range'])]
        if axis < 0:
            nodes += [make_node('Add', [name + '_axis', name + '_dim'], [name + '_true_axis']), make_node('Equal', [name + '_range', name + '_true_axis'], [name + '_one_hot'])]
        else:
            nodes += [make_node('Equal', [name + '_range', name + '_axis'], [name + '_one_hot'])]
        nodes += [make_node('Cast', [name + '_one_hot'], [name + '_one_hot_int'], to=int(TensorProto.INT64)), make_node('Mul', [name + '_repeats', name + '_one_hot_int'], [name + '_mul']), make_node('Add', [name + '_mul', name + '_1'], [name + '_add']), make_node('Concat', [name + '_1', name + '_add'], [name + '_repeats_tensor'], axis=0)]
        if axis == -1:
            nodes += [make_node('Concat', [name + '_shape', name + '_1'], [name + '_unsqueeze_shape'], axis=0), make_node('Reshape', [input_nodes[0], name + '_unsqueeze_shape'], [name + '_unsqueeze'])]
        else:
            nodes += [make_node('Unsqueeze', [input_nodes[0]], [name + '_unsqueeze'], axes=[axis + 1])]
        nodes += [make_node('Tile', [name + '_unsqueeze', name + '_repeats_tensor'], [name + '_tile']), make_node('Mul', [name + '_shape', name + '_add'], [name + '_new_shape']), make_node('Reshape', [name + '_tile', name + '_new_shape'], [name], name=name)]
    return nodes