import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
def transpose_last_two_dim(name, kwargs):
    """Helper function to transpose the last two dims of the input tensor
    """
    from onnx.helper import make_node
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor([1], name + '_1', kwargs['initializer'])
    create_tensor([8], name + '_8', kwargs['initializer'])
    perm = [i for i in range(8)]
    perm[6], perm[7] = (7, 6)
    nodes = [make_node('Shape', [name], [name + '_shape']), make_node('Shape', [name + '_shape'], [name + '_dim']), make_node('Sub', [name + '_8', name + '_dim'], [name + '_sub']), make_node('Concat', [name + '_sub', name + '_0'], [name + '_concat'], axis=0), make_node('Pad', [name + '_shape', name + '_concat', name + '_1'], [name + '_shape_8_dim']), make_node('Reshape', [name, name + '_shape_8_dim'], [name + '_data_8_dim']), make_node('Transpose', [name + '_data_8_dim'], [name + '_data_t'], perm=perm), make_node('Shape', [name + '_data_t'], [name + '_new_shape_']), make_node('Slice', [name + '_new_shape_', name + '_sub', name + '_8', name + '_0'], [name + '_new_shape']), make_node('Reshape', [name + '_data_t', name + '_new_shape'], [name + '_transposed'])]
    return nodes