from typing import Optional
import numpy as np
from onnx import TensorProto
from onnx.helper import np_dtype_to_tensor_dtype
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32
from onnx.reference.custom_element_types import (
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_quantize_linear import reshape_input
def get_x_type(self, x: np.ndarray) -> int:
    tensor_dtype = None
    if x.dtype == float8e4m3fn and x.dtype.descr[0][0] == 'e4m3fn':
        tensor_dtype = TensorProto.FLOAT8E4M3FN
    elif x.dtype == float8e4m3fnuz and x.dtype.descr[0][0] == 'e4m3fnuz':
        tensor_dtype = TensorProto.FLOAT8E4M3FNUZ
    elif x.dtype == float8e5m2 and x.dtype.descr[0][0] == 'e5m2':
        tensor_dtype = TensorProto.FLOAT8E5M2
    elif x.dtype == float8e5m2fnuz and x.dtype.descr[0][0] == 'e5m2fnuz':
        tensor_dtype = TensorProto.FLOAT8E5M2FNUZ
    elif x.dtype == uint4 and x.dtype.descr[0][0] == 'uint4':
        tensor_dtype = TensorProto.UINT4
    elif x.dtype == int4 and x.dtype.descr[0][0] == 'int4':
        tensor_dtype = TensorProto.INT4
    else:
        tensor_dtype = np_dtype_to_tensor_dtype(x.dtype)
    return tensor_dtype