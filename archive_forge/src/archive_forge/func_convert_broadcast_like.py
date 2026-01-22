import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('broadcast_like')
def convert_broadcast_like(node, **kwargs):
    """Map MXNet's broadcast_like operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    lhs = input_nodes[0]
    rhs = input_nodes[1]
    lhs_axes = convert_string_to_list(str(attrs.get('lhs_axes', 'None')))
    rhs_axes = convert_string_to_list(str(attrs.get('rhs_axes', 'None')))
    if lhs_axes[0] is None or rhs_axes[0] is None:
        nodes = [make_node('Shape', [rhs], [name + '_rhs_shape']), make_node('Expand', [lhs, name + '_rhs_shape'], [name], name=name)]
        return nodes
    lhs_axes = [[i] for i in lhs_axes]
    rhs_axes = [[i] for i in rhs_axes]
    create_tensor([0], name + '_0', kwargs['initializer'])
    create_tensor(lhs_axes, name + '_lhs_axes', kwargs['initializer'])
    create_tensor(rhs_axes, name + '_rhs_axes', kwargs['initializer'])
    nodes = [make_node('Shape', [lhs], [name + '_lhs_shape']), make_node('Shape', [rhs], [name + '_rhs_shape']), make_node('Shape', [name + '_lhs_shape'], [name + '_lhs_dim']), make_node('Less', [name + '_lhs_axes', name + '_0'], [name + '_less']), make_node('Cast', [name + '_less'], [name + '_mask'], to=int(onnx.TensorProto.INT64)), make_node('Mul', [name + '_mask', name + '_lhs_dim'], [name + '_mul']), make_node('Add', [name + '_lhs_axes', name + '_mul'], [name + '_lhs_axes_positive']), make_node('GatherND', [name + '_rhs_shape', name + '_rhs_axes'], [name + '_gather']), make_node('ScatterND', [name + '_lhs_shape', name + '_lhs_axes_positive', name + '_gather'], [name + '_scatter']), make_node('Expand', [lhs, name + '_scatter'], [name], name=name)]
    return nodes