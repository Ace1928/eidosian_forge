import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_CenterCropPad(self) -> None:
    input_ = helper.make_tensor('input', TensorProto.FLOAT, dims=[2, 4], vals=np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    shape = helper.make_tensor('shape', TensorProto.INT64, dims=[2], vals=np.array([3, 3]))
    self._test_op_upgrade('CenterCropPad', 18, [[], []], [[3, 3]], [TensorProto.FLOAT, TensorProto.INT64], initializer=[input_, shape])