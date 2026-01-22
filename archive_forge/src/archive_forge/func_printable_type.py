import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def printable_type(t: TypeProto) -> str:
    if t.WhichOneof('value') == 'tensor_type':
        s = TensorProto.DataType.Name(t.tensor_type.elem_type)
        if t.tensor_type.HasField('shape'):
            if len(t.tensor_type.shape.dim):
                s += str(', ' + 'x'.join(map(printable_dim, t.tensor_type.shape.dim)))
            else:
                s += ', scalar'
        return s
    if t.WhichOneof('value') is None:
        return ''
    return f'Unknown type {t.WhichOneof('value')}'