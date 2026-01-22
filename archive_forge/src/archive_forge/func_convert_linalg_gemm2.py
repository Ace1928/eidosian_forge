import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_linalg_gemm2')
def convert_linalg_gemm2(node, **kwargs):
    """Map MXNet's _linalg_gemm2 operator attributes to onnx's
    MatMul and Transpose operators based on the values set for
    transpose_a, transpose_b attributes.
    Return multiple nodes created.
    """
    from onnx.helper import make_node
    name, inputs, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    alpha = float(attrs.get('alpha', 1.0))
    axis = attrs.get('axis', 'None')
    trans_a = get_boolean_attribute_value(attrs, 'transpose_a')
    trans_b = get_boolean_attribute_value(attrs, 'transpose_b')
    if axis != 'None':
        raise NotImplementedError('_linalg_gemm2 does not currently support axis!=None')
    nodes = []
    input_nodes = []
    if trans_a:
        nodes += transpose_last_two_dim(inputs[0], kwargs)
        input_nodes.append(inputs[0] + '_transposed')
    else:
        input_nodes.append(inputs[0])
    if trans_b:
        nodes += transpose_last_two_dim(inputs[1], kwargs)
        input_nodes.append(inputs[1] + '_transposed')
    else:
        input_nodes.append(inputs[1])
    if alpha == 1:
        nodes += [make_node('MatMul', input_nodes, [name])]
        return nodes
    create_const_scalar_node(name + '_alpha', dtype.type(alpha), kwargs)
    nodes += [make_node('MatMul', input_nodes, [name + '_matmul']), make_node('Mul', [name + '_matmul', name + '_alpha'], [name])]
    return nodes