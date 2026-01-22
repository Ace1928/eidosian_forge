import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('stack')
def convert_stack(node, **kwargs):
    """Map MXNet's stack operator to onnx operators.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get('axis', 0))
    idx = 0
    nodes = []
    for input_node in input_nodes:
        nodes.append(onnx.helper.make_node('Unsqueeze', inputs=[input_node], outputs=[name + '_unsqueeze' + str(idx)], axes=[axis]))
        idx += 1
    nodes.append(onnx.helper.make_node('Concat', inputs=[name + '_unsqueeze' + str(i) for i in range(len(nodes))], outputs=[name], name=name, axis=axis))
    return nodes