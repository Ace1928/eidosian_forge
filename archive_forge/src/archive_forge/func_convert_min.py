import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('min')
def convert_min(node, **kwargs):
    """Map MXNet's min operator attributes to onnx's ReduceMin operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    mx_axis = str(attrs.get('axis', 'None'))
    axes = convert_string_to_list(mx_axis) if mx_axis != 'None' else None
    keepdims = get_boolean_attribute_value(attrs, 'keepdims')
    if axes is not None:
        if keepdims:
            node = make_node('ReduceMin', input_nodes, [name], axes=axes, keepdims=keepdims)
            return [node]
        else:
            create_tensor([1], name + '_1', kwargs['initializer'])
            nodes = [make_node('ReduceMin', input_nodes, [name + '_reduce'], axes=axes, keepdims=keepdims), make_node('Shape', [name + '_reduce'], [name + '_reduce_shape']), make_node('Concat', [name + '_1', name + '_reduce_shape'], [name + '_concat'], axis=0), make_node('Reshape', [name + '_reduce', name + '_concat'], [name + '_reshape']), make_node('Squeeze', [name + '_reshape'], [name], axes=[0])]
            return nodes
    elif keepdims:
        node = make_node('ReduceMin', input_nodes, [name], keepdims=keepdims)
        return [node]
    else:
        create_tensor([1], name + '_1', kwargs['initializer'])
        nodes = [make_node('ReduceMin', input_nodes, [name + '_rmin'], keepdims=keepdims), make_node('Reshape', [name + '_rmin', name + '_1'], [name])]
        return nodes