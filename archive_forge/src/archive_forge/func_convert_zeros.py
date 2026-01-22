import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_zeros')
def convert_zeros(node, **kwargs):
    """Map MXNet's zeros operator attributes to onnx's ConstantOfShape operator.
    """
    from onnx.helper import make_node, make_tensor
    name, _, attrs = get_inputs(node, kwargs)
    dtype = attrs.get('dtype')
    data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
    shape = convert_string_to_list(attrs.get('shape'))
    shape = [x if x else 1 for x in shape]
    create_tensor(shape, name + '_shape', kwargs['initializer'])
    tensor_value = make_tensor(name + '_zero', data_type, [1], [0])
    nodes = [make_node('ConstantOfShape', [name + '_shape'], [name], name=name, value=tensor_value)]
    return (nodes, (dtype,))