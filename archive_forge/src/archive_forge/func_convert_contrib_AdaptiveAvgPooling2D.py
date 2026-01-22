import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_contrib_AdaptiveAvgPooling2D')
def convert_contrib_AdaptiveAvgPooling2D(node, **kwargs):
    """Map MXNet's _contrib_AdaptiveAvgPooling2D operator
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    output_size = attrs.get('output_size', '1')
    output_size = convert_string_to_list(output_size)
    if len(output_size) <= 2:
        if output_size[0] != 1 or (len(output_size) == 2 and output_size[1] != 1):
            raise NotImplementedError('_contrib_AdaptiveAvgPooling2D operator with output_size != 1                                 not yet implemented.')
    nodes = [make_node('GlobalAveragePool', [input_nodes[0]], [name], name=name)]
    return nodes