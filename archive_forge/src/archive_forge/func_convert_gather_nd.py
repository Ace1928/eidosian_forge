import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('gather_nd')
def convert_gather_nd(node, **kwargs):
    """Map MXNet's gather_ND operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    name, input_nodes, _ = get_inputs(node, kwargs)
    data = input_nodes[0]
    indices = input_nodes[1]
    perm = [7] + [i for i in range(1, 7)] + [0]
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([1], name + '_1', kwargs['initializer'])
    create_tensor([8], name + '_8', kwargs['initializer'])
    nodes = [make_node('Shape', [indices], [name + '_indices_shape']), make_node('Shape', [name + '_indices_shape'], [name + '_indices_dim']), make_node('Sub', [name + '_8', name + '_indices_dim'], [name + '_sub0_out']), make_node('Concat', [name + '_0', name + '_sub0_out'], [name + '_concat0_out'], axis=0), make_node('Pad', [name + '_indices_shape', name + '_concat0_out', name + '_1'], [name + '_shape_8_dim']), make_node('Reshape', [indices, name + '_shape_8_dim'], [name + '_indices_8_dim']), make_node('Transpose', [name + '_indices_8_dim'], [name + '_transpose0_output'], perm=perm), make_node('Slice', [name + '_indices_shape', name + '_0', name + '_1'], [name + '_slice0_out']), make_node('Slice', [name + '_indices_shape', name + '_1', name + '_indices_dim'], [name + '_slice1_out']), make_node('Concat', [name + '_slice1_out', name + '_slice0_out'], [name + '_concat1_out'], axis=0), make_node('Reshape', [name + '_transpose0_output', name + '_concat1_out'], [name + '_reshape0_out']), make_node('Cast', [name + '_reshape0_out'], [name + '_cast0_out'], to=int(onnx.TensorProto.INT64)), make_node('GatherND', [data, name + '_cast0_out'], [name], name=name)]
    return nodes