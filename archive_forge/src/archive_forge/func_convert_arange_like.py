import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_contrib_arange_like')
def convert_arange_like(node, **kwargs):
    """Map MXNet's arange_like operator attributes to onnx's Range and Reshape operators.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')
    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    axis = attrs.get('axis', 'None')
    start = attrs.get('start', 0.0)
    step = attrs.get('step', 1.0)
    repeat = int(attrs.get('repeat', 1))
    if repeat != 1:
        raise NotImplementedError('arange_like operator with repeat != 1 not yet implemented.')
    create_const_scalar_node(name + '_start', np.dtype(dtype).type(start), kwargs)
    create_const_scalar_node(name + '_step', np.dtype(dtype).type(step), kwargs)
    create_const_scalar_node(name + '_half_step', np.dtype(dtype).type(float(step) * 0.5), kwargs)
    nodes = []
    if axis == 'None':
        nodes += [make_node('Shape', [input_nodes[0]], [name + '_shape0_out']), make_node('ReduceProd', [name + '_shape0_out'], [name + '_redprod0_out']), make_node('Squeeze', [name + '_redprod0_out'], [name + '_reshape0_out'], axes=[0]), make_node('Cast', [name + '_reshape0_out'], [name + '_cast0_out'], to=dtype_t), make_node('Mul', [name + '_cast0_out', name + '_step'], [name + '_mul0_out']), make_node('Add', [name + '_mul0_out', name + '_start'], [name + '_add1_out']), make_node('Sub', [name + '_add1_out', name + '_half_step'], [name + '_sub0_out']), make_node('Range', [name + '_start', name + '_sub0_out', name + '_step'], [name + '_range0_out']), make_node('Reshape', [name + '_range0_out', name + '_shape0_out'], [name], name=name)]
    else:
        create_tensor([int(axis)], name + '_axis_start', kwargs['initializer'], dtype='int64')
        create_tensor([int(axis) + 1], name + '_axis_end', kwargs['initializer'], dtype='int64')
        nodes += [make_node('Shape', [input_nodes[0]], [name + '_shape0_out']), make_node('Slice', [name + '_shape0_out', name + '_axis_start', name + '_axis_end'], [name + '_slice0_out']), make_node('ReduceProd', [name + '_slice0_out'], [name + '_reprod0_out']), make_node('Squeeze', [name + '_reprod0_out'], [name + '_reshape0_out'], axes=[0]), make_node('Cast', [name + '_reshape0_out'], [name + '_cast0_out'], to=dtype_t), make_node('Mul', [name + '_cast0_out', name + '_step'], [name + '_mul0_out']), make_node('Add', [name + '_mul0_out', name + '_start'], [name + '_add1_out']), make_node('Sub', [name + '_add1_out', name + '_half_step'], [name + '_sub0_out']), make_node('Range', [name + '_start', name + '_sub0_out', name + '_step'], [name], name=name)]
    return nodes