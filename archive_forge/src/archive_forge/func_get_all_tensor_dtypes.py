import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def get_all_tensor_dtypes() -> KeysView[int]:
    """Get all tensor types from TensorProto.

    Returns:
        all tensor types from TensorProto
    """
    return mapping.TENSOR_TYPE_MAP.keys()