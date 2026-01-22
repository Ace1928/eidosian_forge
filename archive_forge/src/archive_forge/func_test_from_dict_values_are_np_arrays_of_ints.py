import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_from_dict_values_are_np_arrays_of_ints(self):
    zero_array = np.array([1, 2])
    one_array = np.array([9, 10])
    map_proto = numpy_helper.from_dict({0: zero_array, 1: one_array})
    self.assertIsInstance(map_proto, onnx.MapProto)
    out_tensor = numpy_helper.to_array(map_proto.values.tensor_values[0])
    self.assertEqual(out_tensor[0], zero_array[0])
    self.assertEqual(out_tensor[1], zero_array[1])
    out_tensor = numpy_helper.to_array(map_proto.values.tensor_values[1])
    self.assertEqual(out_tensor[0], one_array[0])
    self.assertEqual(out_tensor[1], one_array[1])