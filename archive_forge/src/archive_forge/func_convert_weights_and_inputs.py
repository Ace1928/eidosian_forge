import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
@mx_op.register('null')
def convert_weights_and_inputs(node, **kwargs):
    """Helper function to convert weights and inputs.
    """
    name, _, _ = get_inputs(node, kwargs)
    if kwargs['is_input'] is False:
        weights = kwargs['weights']
        initializer = kwargs['initializer']
        np_arr = weights[name]
        data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np_arr.dtype]
        dims = np.shape(np_arr)
        tensor_node = onnx.helper.make_tensor_value_info(name, data_type, dims)
        initializer.append(onnx.helper.make_tensor(name=name, data_type=data_type, dims=dims, vals=np_arr.flatten().tolist(), raw=False))
        return ([tensor_node], (np_arr.dtype,))
    else:
        dtype_t = kwargs['in_type']
        dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[dtype_t]
        tval_node = onnx.helper.make_tensor_value_info(name, dtype_t, kwargs['in_shape'])
        return ([tval_node], (dtype,))