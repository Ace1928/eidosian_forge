import os
import tempfile
import unittest
import numpy as np
import numpy.testing as npt
import onnx
import onnx.helper
import onnx.model_container
import onnx.numpy_helper
import onnx.reference
def common_check_reference_evaluator(self, container):
    X = np.arange(9).astype(np.float32).reshape((-1, 3))
    ref = onnx.reference.ReferenceEvaluator(container)
    got = ref.run(None, {'X': X})
    expected = np.array([[945000, 1015200, 1085400], [2905200, 3121200, 3337200], [4865400, 5227200, 5589000]], dtype=np.float32)
    npt.assert_allclose(expected, got[0])