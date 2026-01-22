import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('Reshape')
def convert_reshape(node, **kwargs):
    """Map MXNet's Reshape operator attributes to onnx's Reshape operator.
    Converts output shape attribute to output shape tensor
    and return multiple created nodes.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    reverse = attrs.get('reverse', 'False')
    targ_shape = convert_string_to_list(attrs['shape'])
    if -4 in targ_shape and -3 not in targ_shape and (-2 not in targ_shape) and (reverse != 'True'):
        if 0 not in targ_shape:
            targ_shape = [i for i in targ_shape if i != -4]
        else:
            ind_4 = targ_shape.index(-4)
            ind0 = len(targ_shape) - 1 - targ_shape[::-1].index(0)
            if ind_4 > ind0:
                targ_shape = [i for i in targ_shape if i != -4]
    if targ_shape == [-3, 0] and reverse != 'True':
        targ_shape = [-1, 0]
        reverse = 'True'
    special_case = False
    if targ_shape == [0, 0, -3, -3] and reverse != 'True':
        special_case = True
        nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Split', [name + '_shape'], [name + '_dim0', name + '_dim1', name + '_dim2', name + '_dim3', name + '_dim4', name + '_dim5'], axis=0), make_node('Mul', [name + '_dim2', name + '_dim3'], [name + '_mul_1']), make_node('Mul', [name + '_dim4', name + '_dim5'], [name + '_mul_2']), make_node('Concat', [name + '_dim0', name + '_dim1', name + '_mul_1', name + '_mul_2'], [name + '_shape_new'], axis=0), make_node('Reshape', [input_nodes[0], name + '_shape_new'], [name], name=name)]
    if targ_shape == [0, -4, -1, 4, 0, 0] and reverse != 'True':
        special_case = True
        create_tensor([4], name + '_4', kwargs['initializer'])
        nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Split', [name + '_shape'], [name + '_dim0', name + '_dim1', name + '_dim2', name + '_dim3'], axis=0), make_node('Div', [name + '_dim1', name + '_4'], [name + '_div']), make_node('Concat', [name + '_dim0', name + '_div', name + '_4', name + '_dim2', name + '_dim3'], [name + '_shape_new'], axis=0), make_node('Reshape', [input_nodes[0], name + '_shape_new'], [name], name=name)]
    if targ_shape == [0, 0, -4, 2, 2, 0, 0] and reverse != 'True':
        special_case = True
        create_tensor([2], name + '_2', kwargs['initializer'])
        nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Split', [name + '_shape'], [name + '_dim0', name + '_dim1', name + '_dim2', name + '_dim3', name + '_dim4'], axis=0), make_node('Concat', [name + '_dim0', name + '_dim1', name + '_2', name + '_2', name + '_dim3', name + '_dim4'], [name + '_shape_new'], axis=0), make_node('Reshape', [input_nodes[0], name + '_shape_new'], [name], name=name)]
    if targ_shape == [-4, 1, -1, 0, 0, 0] and reverse != 'True':
        special_case = True
        create_tensor([1], name + '_1', kwargs['initializer'])
        create_tensor([-1], name + '_m1', kwargs['initializer'])
        nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Split', [name + '_shape'], [name + '_dim0', name + '_dim1', name + '_dim2', name + '_dim3'], axis=0), make_node('Concat', [name + '_1', name + '_m1', name + '_dim1', name + '_dim2', name + '_dim3'], [name + '_shape_new'], axis=0), make_node('Reshape', [input_nodes[0], name + '_shape_new'], [name], name=name)]
    if targ_shape == [-4, 1, 1000, 0, 0] and reverse != 'True':
        special_case = True
        create_tensor([1], name + '_1', kwargs['initializer'])
        create_tensor([1000], name + '_1000', kwargs['initializer'])
        nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Split', [name + '_shape'], [name + '_dim0', name + '_dim1', name + '_dim2'], axis=0), make_node('Concat', [name + '_1', name + '_1000', name + '_dim1', name + '_dim2'], [name + '_shape_new'], axis=0), make_node('Reshape', [input_nodes[0], name + '_shape_new'], [name], name=name)]
    if targ_shape == [0, -4, 12, -1, 0] and reverse != 'True':
        special_case = True
        create_tensor([-1], name + '_m1', kwargs['initializer'])
        create_tensor([12], name + '_12', kwargs['initializer'])
        nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Split', [name + '_shape'], [name + '_dim0', name + '_dim1', name + '_dim2'], axis=0), make_node('Concat', [name + '_dim0', name + '_12', name + '_m1', name + '_dim2'], [name + '_shape_new'], axis=0), make_node('Reshape', [input_nodes[0], name + '_shape_new'], [name], name=name)]
    if targ_shape == [0, -4, 16, -1, 0] and reverse != 'True':
        special_case = True
        create_tensor([-1], name + '_m1', kwargs['initializer'])
        create_tensor([16], name + '_16', kwargs['initializer'])
        nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Split', [name + '_shape'], [name + '_dim0', name + '_dim1', name + '_dim2'], axis=0), make_node('Concat', [name + '_dim0', name + '_16', name + '_m1', name + '_dim2'], [name + '_shape_new'], axis=0), make_node('Reshape', [input_nodes[0], name + '_shape_new'], [name], name=name)]
    if targ_shape == [-3, -1] and reverse != 'True':
        special_case = True
        create_tensor([0], name + '_0', kwargs['initializer'])
        create_tensor([1], name + '_1', kwargs['initializer'])
        create_tensor([2], name + '_2', kwargs['initializer'])
        create_tensor([-1], name + '_-1', kwargs['initializer'])
        nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Slice', [name + '_shape', name + '_0', name + '_1'], [name + '_1st_dim']), make_node('Slice', [name + '_shape', name + '_1', name + '_2'], [name + '_2nd_dim']), make_node('Mul', [name + '_1st_dim', name + '_2nd_dim'], [name + '_mul']), make_node('Concat', [name + '_mul', name + '_-1'], [name + '_shape_new'], axis=0), make_node('Reshape', [input_nodes[0], name + '_shape_new'], [name], name=name)]
    if special_case:
        return nodes
    not_supported_shape = [-2, -3, -4]
    for val in targ_shape:
        if val in not_supported_shape:
            raise AttributeError('Reshape: Shape value not supported in ONNX', val)
    create_tensor(targ_shape, name + '_targ_shape', kwargs['initializer'])
    nodes = []
    if reverse == 'False':
        nodes += [make_node('Reshape', [input_nodes[0], name + '_targ_shape'], [name], name=name)]
    else:
        create_tensor([0], name + '_0', kwargs['initializer'])
        create_tensor([1], name + '_1', kwargs['initializer'])
        nodes += [make_node('Shape', [name + '_targ_shape'], [name + '_targ_dim']), make_node('Shape', [input_nodes[0]], [name + '_orig_shape']), make_node('Shape', [name + '_orig_shape'], [name + '_orig_dim']), make_node('Sub', [name + '_targ_dim', name + '_orig_dim'], [name + '_dim_diff']), make_node('Abs', [name + '_dim_diff'], [name + '_pad_len']), make_node('Less', [name + '_targ_dim', name + '_orig_dim'], [name + '_targ_less_orig']), make_node('Less', [name + '_orig_dim', name + '_targ_dim'], [name + '_orig_less_targ']), make_node('Where', [name + '_targ_less_orig', name + '_pad_len', name + '_0'], [name + '_targ_pad_len']), make_node('Where', [name + '_orig_less_targ', name + '_pad_len', name + '_0'], [name + '_orig_pad_len']), make_node('Concat', [name + '_targ_pad_len', name + '_0'], [name + '_targ_pads'], axis=0), make_node('Concat', [name + '_orig_pad_len', name + '_0'], [name + '_orig_pads'], axis=0), make_node('Pad', [name + '_targ_shape', name + '_targ_pads', name + '_1'], [name + '_targ_shape_padded'], mode='constant'), make_node('Pad', [name + '_orig_shape', name + '_orig_pads', name + '_1'], [name + '_orig_shape_padded'], mode='constant'), make_node('Equal', [name + '_targ_shape_padded', name + '_0'], [name + '_targ_shape_0_mask']), make_node('Where', [name + '_targ_shape_0_mask', name + '_orig_shape_padded', name + '_targ_shape_padded'], [name + '_targ_shape_new']), make_node('Shape', [name + '_targ_shape_new'], [name + '_targ_new_dim']), make_node('Slice', [name + '_targ_shape_new', name + '_targ_pad_len', name + '_targ_new_dim'], [name + '_targ_shape_final']), make_node('Reshape', [input_nodes[0], name + '_targ_shape_final'], [name], name=name)]
    return nodes