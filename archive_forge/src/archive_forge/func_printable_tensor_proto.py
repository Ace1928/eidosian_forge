import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def printable_tensor_proto(t: TensorProto) -> str:
    s = f'%{t.name}['
    s += TensorProto.DataType.Name(t.data_type)
    if t.dims is not None:
        if len(t.dims):
            s += str(', ' + 'x'.join(map(str, t.dims)))
        else:
            s += ', scalar'
    s += ']'
    return s