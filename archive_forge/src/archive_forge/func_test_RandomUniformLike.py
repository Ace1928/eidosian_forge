import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_RandomUniformLike(self) -> None:
    like = helper.make_tensor('a', TensorProto.FLOAT, dims=[3, 4, 5], vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(), raw=True)
    self._test_op_upgrade('RandomUniformLike', 1, [[3, 4, 5]], [[3, 4, 5]], initializer=[like])