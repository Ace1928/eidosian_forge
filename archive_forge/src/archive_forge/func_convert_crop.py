import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('Crop')
def convert_crop(node, **kwargs):
    """Map MXNet's crop operator attributes to onnx's Slice operator
    """
    from onnx.helper import make_node
    name, inputs, attrs = get_inputs(node, kwargs)
    num_inputs = len(inputs)
    y, x = convert_string_to_list(attrs.get('offset', '(0, 0)'))
    h, w = convert_string_to_list(attrs.get('h_w', '(0, 0)'))
    center_crop = attrs.get('center_crop', 'False')
    if center_crop in ['True', '1']:
        raise NotImplementedError('Crop does not currently support center_crop==True')
    nodes = []
    create_tensor([y, x], name + '_starts', kwargs['initializer'])
    create_tensor([2, 3], name + '_axes', kwargs['initializer'])
    if num_inputs == 1:
        create_tensor([y + h, x + w], name + '_ends', kwargs['initializer'])
    else:
        create_tensor([0], name + '_0', kwargs['initializer'])
        create_tensor([2], name + '_2', kwargs['initializer'])
        create_tensor([4], name + '_4', kwargs['initializer'])
        nodes += [make_node('Shape', [inputs[1]], [name + '_shape']), make_node('Slice', [name + '_shape', name + '_2', name + '_4', name + '_0'], [name + '_h_w']), make_node('Add', [name + '_starts', name + '_h_w'], [name + '_ends'])]
    nodes += [make_node('Slice', [inputs[0], name + '_starts', name + '_ends', name + '_axes'], [name])]
    return nodes