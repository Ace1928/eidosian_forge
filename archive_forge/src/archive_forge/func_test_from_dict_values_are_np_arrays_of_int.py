import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_from_dict_values_are_np_arrays_of_int(self):
    map_proto = numpy_helper.from_dict({0: np.array(1), 1: np.array(9)})
    self.assertIsInstance(map_proto, onnx.MapProto)
    self.assertEqual(numpy_helper.to_array(map_proto.values.tensor_values[0]), np.array(1))
    self.assertEqual(numpy_helper.to_array(map_proto.values.tensor_values[1]), np.array(9))