import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('_sample_multinomial')
def convert_multinomial(node, **kwargs):
    """Map MXNet's multinomial operator attributes to onnx's
    Multinomial operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(attrs.get('dtype', 'int32'))]
    sample_size = convert_string_to_list(attrs.get('shape', '1'))
    if len(sample_size) < 2:
        sample_size = sample_size[-1]
    else:
        raise AttributeError('ONNX currently supports integer sample_size only')
    node = onnx.helper.make_node('Multinomial', input_nodes, [name], dtype=dtype, sample_size=sample_size, name=name)
    return [node]