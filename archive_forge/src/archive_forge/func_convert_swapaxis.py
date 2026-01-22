import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('SwapAxis')
def convert_swapaxis(node, **kwargs):
    """Map MXNet's SwapAxis operator
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    dim1 = int(attrs.get('dim1', '0'))
    dim2 = int(attrs.get('dim2', '0'))
    if dim1 < 0 or dim2 < 0:
        raise NotImplementedError('SwapAxis conversion does not support dim1 < 0                                   or dim2 < 0')
    indices = [[dim1], [dim2]]
    vals = [dim2, dim1]
    perm = [i for i in range(8)]
    perm[dim1], perm[dim2] = (dim2, dim1)
    create_tensor(indices, name + '_ind', kwargs['initializer'])
    create_tensor(indices[::-1], name + '_ind_rev', kwargs['initializer'])
    create_tensor(vals, name + '_vals', kwargs['initializer'])
    create_tensor(perm, name + '_perm', kwargs['initializer'])
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([1], name + '_1', kwargs['initializer'])
    create_tensor([8], name + '_8', kwargs['initializer'])
    nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Shape', [name + '_shape'], [name + '_dim']), make_node('Sub', [name + '_8', name + '_dim'], [name + '_sub']), make_node('ScatterND', [name + '_perm', name + '_ind', name + '_vals'], [name + '_perm_new']), make_node('GatherND', [name + '_shape', name + '_ind'], [name + '_gather']), make_node('ScatterND', [name + '_shape', name + '_ind_rev', name + '_gather'], [name + '_shape_new']), make_node('Concat', [name + '_0', name + '_sub'], [name + '_pad'], axis=0), make_node('Pad', [name + '_shape', name + '_pad', name + '_1'], [name + '_shape_padded']), make_node('Reshape', [input_nodes[0], name + '_shape_padded'], [name + '_data_padded']), make_node('Transpose', [name + '_data_padded'], [name + '_trans'], perm=perm), make_node('Reshape', [name + '_trans', name + '_shape_new'], [name])]
    return nodes