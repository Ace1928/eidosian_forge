import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('argmin')
def convert_argmin(node, **kwargs):
    """Map MXNet's argmin operator attributes to onnx's ArgMin operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = str(attrs.get('axis', 'None'))
    keepdims = get_boolean_attribute_value(attrs, 'keepdims')
    input_dtype = get_input_dtypes(node, kwargs)[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input_dtype]
    if axis == 'None':
        create_tensor([-1], name + '_-1', kwargs['initializer'])
        if keepdims:
            create_tensor([1], name + '_1', kwargs['initializer'])
            nodes = [make_node('Shape', [input_nodes[0]], [name + '_shape']), make_node('Shape', [name + '_shape'], [name + '_dim']), make_node('Tile', [name + '_1', name + '_dim'], [name + '_tile']), make_node('Reshape', [input_nodes[0], name + '_-1'], [name + '_reshape']), make_node('ArgMin', [name + '_reshape'], [name + '_argmin'], axis=0, keepdims=True), make_node('Reshape', [name + '_argmin', name + '_tile'], [name + '_ret']), make_node('Cast', [name + '_ret'], [name], to=dtype_t, name=name)]
        else:
            nodes = [make_node('Reshape', [input_nodes[0], name + '_-1'], [name + '_reshape']), make_node('ArgMin', [name + '_reshape'], [name + '_argmin'], axis=0, keepdims=True), make_node('Cast', [name + '_argmin'], [name], to=dtype_t, name=name)]
    else:
        axis = int(axis)
        nodes = [make_node('ArgMin', [input_nodes[0]], [name + '_argmin'], axis=axis, keepdims=keepdims), make_node('Cast', [name + '_argmin'], [name], to=dtype_t, name=name)]
    return nodes