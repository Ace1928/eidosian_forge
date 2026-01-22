import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('slice')
def convert_slice(node, **kwargs):
    """Map MXNet's slice operator to onnx Slice operator."""
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    starts = convert_string_to_list(attrs.get('begin'))
    ends = convert_string_to_list(attrs.get('end'))
    steps = convert_string_to_list(attrs.get('step', '[]'))
    assert len(starts) == len(ends)
    if len(steps) == 0 or (len(steps) == 1 and steps[0] is None):
        steps = [1 for x in starts]
    else:
        assert len(steps) == len(starts)
    steps = [1 if x is None else x for x in steps]
    for i, s in enumerate(steps):
        if s < 0:
            raise NotImplementedError('slice operator does not support negative steps yet')
        if starts[i] is None:
            starts[i] = 0
        if ends[i] is None:
            ends[i] = 2 ** 63 - 1
    axes = [i for i in range(len(starts))]
    create_tensor(axes, name + '_axes', kwargs['initializer'])
    create_tensor(starts, name + '_starts', kwargs['initializer'])
    create_tensor(ends, name + '_ends', kwargs['initializer'])
    create_tensor(steps, name + '_steps', kwargs['initializer'])
    nodes = [make_node('Slice', [input_nodes[0], name + '_starts', name + '_ends', name + '_axes', name + '_steps'], [name], name=name)]
    return nodes