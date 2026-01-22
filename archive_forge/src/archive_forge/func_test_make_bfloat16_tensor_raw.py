import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
def test_make_bfloat16_tensor_raw(self) -> None:
    np_array = np.array([[1.0, 2.0], [3.0, 4.0], [0.099853515625, 0.099365234375], [0.0998535081744, 0.1], [np.nan, np.inf]], dtype=np.float32)
    np_results = np.array([[struct.unpack('!f', bytes.fromhex('3F800000'))[0], struct.unpack('!f', bytes.fromhex('40000000'))[0]], [struct.unpack('!f', bytes.fromhex('40400000'))[0], struct.unpack('!f', bytes.fromhex('40800000'))[0]], [struct.unpack('!f', bytes.fromhex('3DCC0000'))[0], struct.unpack('!f', bytes.fromhex('3DCB0000'))[0]], [struct.unpack('!f', bytes.fromhex('3DCC0000'))[0], struct.unpack('!f', bytes.fromhex('3DCC0000'))[0]], [struct.unpack('!f', bytes.fromhex('7FC00000'))[0], struct.unpack('!f', bytes.fromhex('7F800000'))[0]]])

    def truncate(x):
        return x >> 16
    values_as_ints = np_array.astype(np.float32).view(np.uint32).flatten()
    packed_values = truncate(values_as_ints).astype(np.uint16).tobytes()
    tensor = helper.make_tensor(name='test', data_type=TensorProto.BFLOAT16, dims=np_array.shape, vals=packed_values, raw=True)
    self.assertEqual(tensor.name, 'test')
    np.testing.assert_equal(np_results, numpy_helper.to_array(tensor))