import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def tensor_dtype_to_field(tensor_dtype: int) -> str:
    """Convert a TensorProto's data_type to corresponding field name for storage. It can be used while making tensors.

    Args:
        tensor_dtype: TensorProto's data_type

    Returns:
        field name
    """
    return mapping._STORAGE_TENSOR_TYPE_TO_FIELD[mapping.TENSOR_TYPE_MAP[tensor_dtype].storage_dtype]