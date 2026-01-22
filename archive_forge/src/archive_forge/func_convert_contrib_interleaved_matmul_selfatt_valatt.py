import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_contrib_interleaved_matmul_selfatt_valatt')
def convert_contrib_interleaved_matmul_selfatt_valatt(node, **kwargs):
    """Map MXNet's _contrib_interleaved_matmul_selfatt_valatt operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    qkv = input_nodes[0]
    att = input_nodes[1]
    num_heads = int(attrs.get('heads'))
    create_tensor([num_heads], name + '_const_num_heads', kwargs['initializer'])
    create_tensor([0], name + '_const_0', kwargs['initializer'])
    create_tensor([1], name + '_const_1', kwargs['initializer'])
    create_tensor([2], name + '_const_2', kwargs['initializer'])
    create_tensor([3], name + '_const_3', kwargs['initializer'])
    create_tensor([4], name + '_const_4', kwargs['initializer'])
    create_tensor([5], name + '_const_5', kwargs['initializer'])
    create_tensor([0, 0, num_heads, 3, -1], name + '_reshape0_shape', kwargs['initializer'])
    create_tensor([0, 0, 0, 2, 0], name + '_slice_start', kwargs['initializer'])
    create_tensor([0, 0, 0, -1], name + '_reshape1_shape', kwargs['initializer'])
    create_tensor([0, 0, -1], name + '_reshape4_shape', kwargs['initializer'])
    nodes = [make_node('Shape', [qkv], [name + '_shape_qkv']), make_node('Slice', [name + '_shape_qkv', name + '_const_0', name + '_const_1'], [name + '_qkv_d0']), make_node('Slice', [name + '_shape_qkv', name + '_const_1', name + '_const_2'], [name + '_qkv_d1']), make_node('Slice', [name + '_shape_qkv', name + '_const_2', name + '_const_3'], [name + '_qkv_d2']), make_node('Mul', [name + '_qkv_d1', name + '_const_num_heads'], [name + '_mul_out']), make_node('Reshape', [qkv, name + '_reshape0_shape'], [name + '_reshape0_output']), make_node('Shape', [name + '_reshape0_output'], [name + '_shape_reshape0']), make_node('Slice', [name + '_shape_reshape0', name + '_const_4', name + '_const_5'], [name + '_d4']), make_node('Concat', [name + '_mul_out', name + '_qkv_d0', name + '_d4'], [name + '_reshape2_shape'], axis=0), make_node('Concat', [name + '_qkv_d1', name + '_const_num_heads', name + '_qkv_d0', name + '_d4'], [name + '_reshape3_shape'], axis=0), make_node('Concat', [name + '_qkv_d0', name + '_qkv_d1', name + '_qkv_d2', name + '_const_3', name + '_d4'], [name + '_slice_end'], axis=0), make_node('Slice', [name + '_reshape0_output', name + '_slice_start', name + '_slice_end'], [name + '_slice_output']), make_node('Reshape', [name + '_slice_output', name + '_reshape1_shape'], [name + '_reshape1_output']), make_node('Transpose', [name + '_reshape1_output'], [name + '_transpose0_output'], perm=[1, 2, 0, 3]), make_node('Reshape', [name + '_transpose0_output', name + '_reshape2_shape'], [name + '_reshape2_output']), make_node('MatMul', [att, name + '_reshape2_output'], [name + '_matmul_output']), make_node('Reshape', [name + '_matmul_output', name + '_reshape3_shape'], [name + '_reshape3_output']), make_node('Transpose', [name + '_reshape3_output'], [name + '_transpose2_output'], perm=[2, 0, 1, 3]), make_node('Reshape', [name + '_transpose2_output', name + '_reshape4_shape'], [name], name=name)]
    return nodes