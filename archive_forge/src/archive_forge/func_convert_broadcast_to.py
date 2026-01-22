import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('broadcast_to')
def convert_broadcast_to(node, **kwargs):
    """Map MXNet's broadcast_to operator attributes to onnx's Expand
    operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    shape_list = convert_string_to_list(attrs['shape'])
    initializer = kwargs['initializer']
    output_shape_np = np.array(shape_list, dtype='int64')
    data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[output_shape_np.dtype]
    dims = np.shape(output_shape_np)
    output_shape_name = 'expand_attr_tensor' + str(kwargs['idx'])
    tensor_node = onnx.helper.make_tensor_value_info(output_shape_name, data_type, dims)
    initializer.append(onnx.helper.make_tensor(name=output_shape_name, data_type=data_type, dims=dims, vals=shape_list, raw=False))
    input_nodes.append(output_shape_name)
    expand_node = onnx.helper.make_node('Expand', input_nodes, [name], name=name)
    return [tensor_node, expand_node]