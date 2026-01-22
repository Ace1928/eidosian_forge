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
def test_svm_classifier_binary_noprob_linear_sv(self):
    x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
    expected_post = {'NONE': (np.array([0, 0, 0], dtype=np.int64), np.array([[-2.662655, 2.662655], [-2.21481, 2.21481], [-1.766964, 1.766964]], dtype=np.float32)), 'LOGISTIC': (np.array([0, 0, 0], dtype=np.int64), np.array([[0.065213, 0.934787], [0.098428, 0.901572], [0.14592, 0.85408]], dtype=np.float32)), 'SOFTMAX': (np.array([0, 0, 0], dtype=np.int64), np.array([[0.004843, 0.995157], [0.011779, 0.988221], [0.028362, 0.971638]], dtype=np.float32)), 'SOFTMAX_ZERO': (np.array([0, 0, 0], dtype=np.int64), np.array([[0.004843, 0.995157], [0.011779, 0.988221], [0.028362, 0.971638]], dtype=np.float32))}
    for post, expected in expected_post.items():
        with self.subTest(post_transform=post):
            onx = self._get_test_svm_classifier_linear_sv(post, probability=False)
            if post not in {'LOGISTIC', 'SOFTMAX', 'SOFTMAX_ZERO'}:
                self._check_ort(onx, {'X': x}, rev=True, atol=1e-05)
            sess = ReferenceEvaluator(onx)
            got = sess.run(None, {'X': x})
            assert_allclose(expected[1], got[1], atol=1e-06)
            assert_allclose(expected[0], got[0])