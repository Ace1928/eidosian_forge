import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('take')
def convert_take(node, **kwargs):
    """Map MXNet's Take operator attributes to onnx's Gather operator.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get('axis', 0))
    mode = str(attrs.get('mode', 'clip'))
    data = input_nodes[0]
    indices = input_nodes[1]
    nodes = [make_node('Cast', [indices], [name + '_indices'], to=int(TensorProto.INT64))]
    if mode == 'raise':
        nodes += [make_node('Gather', [data, name + '_indices'], [name], axis=axis, name=name)]
        return nodes
    create_tensor([-1], name + '_-1', kwargs['initializer'])
    nodes += [make_node('Shape', [data], [name + '_data_shape'])]
    if axis == -1:
        nodes += [make_node('Shape', [name + '_data_shape'], [name + '_data_dim']), make_node('Add', [name + '_data_dim', name + '_-1'], [name + '_axis_max']), make_node('Slice', [name + '_data_shape', name + '_axis_max', name + '_data_dim'], [name + '_slice0_out'])]
    else:
        create_tensor([axis], name + '_axis', kwargs['initializer'])
        create_tensor([axis + 1], name + '_axis+1', kwargs['initializer'])
        nodes += [make_node('Slice', [name + '_data_shape', name + '_axis', name + '_axis+1'], [name + '_slice0_out'])]
    if mode == 'clip':
        create_tensor([0], name + '_0', kwargs['initializer'])
        nodes += [make_node('Add', [name + '_slice0_out', name + '_-1'], [name + '_max']), make_node('Greater', [name + '_indices', name + '_max'], [name + '_max_mask']), make_node('Where', [name + '_max_mask', name + '_max', name + '_indices'], [name + '_where0_out']), make_node('Less', [name + '_indices', name + '_0'], [name + '_min_mask']), make_node('Where', [name + '_min_mask', name + '_0', name + '_where0_out'], [name + '_where1_out']), make_node('Gather', [data, name + '_where1_out'], [name], axis=axis, name=name)]
    elif mode == 'wrap':
        nodes += [make_node('Mod', [name + '_indices', name + '_slice0_out'], [name + '_mod0_out']), make_node('Gather', [data, name + '_mod0_out'], [name], axis=axis, name=name)]
    else:
        raise NotImplementedError('mode must be clip, wrap or raise.')
    return nodes