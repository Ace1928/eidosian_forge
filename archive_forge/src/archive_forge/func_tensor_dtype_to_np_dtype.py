import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def tensor_dtype_to_np_dtype(tensor_dtype: int) -> np.dtype:
    """Convert a TensorProto's data_type to corresponding numpy dtype. It can be used while making tensor.

    Args:
        tensor_dtype: TensorProto's data_type

    Returns:
        numpy's data_type
    """
    return mapping.TENSOR_TYPE_MAP[tensor_dtype].np_dtype