import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('square')
def convert_square(node, **kwargs):
    """Map MXNet's square operator attributes to onnx's Pow operator
    and return the created node.
    """
    name, input_nodes, _ = get_inputs(node, kwargs)
    initializer = kwargs['initializer']
    data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')]
    power2_name = 'square_tensor' + str(kwargs['idx'])
    tensor_node = onnx.helper.make_tensor_value_info(power2_name, data_type, (1,))
    initializer.append(onnx.helper.make_tensor(name=power2_name, data_type=data_type, dims=(1,), vals=[2], raw=False))
    input_nodes.append(power2_name)
    node = onnx.helper.make_node('Pow', input_nodes, [name], name=name)
    return [tensor_node, node]