import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('sum')
@mx_op.register('sum_axis')
def convert_sum(node, **kwargs):
    """Map MXNet's sum operator attributes to onnx's ReduceSum operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    mx_axis = attrs.get('axis', None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis is not None else None
    keepdims = get_boolean_attribute_value(attrs, 'keepdims')
    if axes:
        node = onnx.helper.make_node('ReduceSum', inputs=input_nodes, outputs=[name], axes=axes, keepdims=keepdims, name=name)
    else:
        node = onnx.helper.make_node('ReduceSum', inputs=input_nodes, outputs=[name], keepdims=keepdims, name=name)
    return [node]