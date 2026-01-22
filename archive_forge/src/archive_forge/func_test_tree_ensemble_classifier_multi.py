import itertools
import unittest
from functools import wraps
from os import getenv
import numpy as np  # type: ignore
from numpy.testing import assert_allclose  # type: ignore
from parameterized import parameterized
import onnx
from onnx import ONNX_ML, TensorProto, TypeProto, ValueInfoProto
from onnx.checker import check_model
from onnx.defs import onnx_ml_opset_version, onnx_opset_version
from onnx.helper import (
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.aionnxml.op_tree_ensemble import (
@unittest.skipIf(not ONNX_ML, reason='onnx not compiled with ai.onnx.ml')
def test_tree_ensemble_classifier_multi(self):
    x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
    expected_post = {'NONE': (np.array([0, 0, 1], dtype=np.int64), np.array([[0.916667, 0.0, 0.083333], [0.569608, 0.191176, 0.239216], [0.302941, 0.431176, 0.265882]], dtype=np.float32)), 'LOGISTIC': (np.array([0, 0, 1], dtype=np.int64), np.array([[0.714362, 0.5, 0.520821], [0.638673, 0.547649, 0.55952], [0.575161, 0.606155, 0.566082]], dtype=np.float32)), 'SOFTMAX': (np.array([0, 0, 1], dtype=np.int64), np.array([[0.545123, 0.217967, 0.23691], [0.416047, 0.284965, 0.298988], [0.322535, 0.366664, 0.310801]], dtype=np.float32)), 'SOFTMAX_ZERO': (np.array([0, 0, 1], dtype=np.int64), np.array([[0.697059, 0.0, 0.302941], [0.416047, 0.284965, 0.298988], [0.322535, 0.366664, 0.310801]], dtype=np.float32)), 'PROBIT': (np.array([0, 0, 1], dtype=np.int64), np.array([[1.383104, 0, -1.383105], [0.175378, -0.873713, -0.708922], [-0.516003, -0.173382, -0.625385]], dtype=np.float32))}
    for post, expected in expected_post.items():
        with self.subTest(post_transform=post):
            onx = self._get_test_tree_ensemble_classifier_multi(post)
            if post != 'PROBIT':
                self._check_ort(onx, {'X': x}, atol=1e-05)
            sess = ReferenceEvaluator(onx)
            got = sess.run(None, {'X': x})
            assert_allclose(expected[1], got[1], atol=1e-06)
            assert_allclose(expected[0], got[0])