import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def tensor_dtype_to_string(tensor_dtype: int) -> str:
    """Get the name of given TensorProto's data_type.

    Args:
        tensor_dtype: TensorProto's data_type

    Returns:
        the name of data_type
    """
    return mapping.TENSOR_TYPE_MAP[tensor_dtype].name