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
def test_tree_ensemble_classifier_binary(self):
    x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
    expected_post = {'NONE': (np.array([0, 1, 1], dtype=np.int64), np.array([[1.0, 0.0], [0.394958, 0.605042], [0.394958, 0.605042]], dtype=np.float32)), 'LOGISTIC': (np.array([0, 1, 1], dtype=np.int64), np.array([[0.5, 0.5], [0.353191, 0.646809], [0.353191, 0.646809]], dtype=np.float32)), 'SOFTMAX': (np.array([0, 1, 1], dtype=np.int64), np.array([[0.5, 0.5], [0.229686, 0.770314], [0.229686, 0.770314]], dtype=np.float32)), 'SOFTMAX_ZERO': (np.array([0, 1, 1], dtype=np.int64), np.array([[0.5, 0.5], [0.229686, 0.770314], [0.229686, 0.770314]], dtype=np.float32)), 'PROBIT': (np.array([0, 1, 1], dtype=np.int64), np.array([[0.0, 0.0], [-0.266426, 0.266426], [-0.266426, 0.266426]], dtype=np.float32))}
    for post, expected in expected_post.items():
        with self.subTest(post_transform=post):
            onx = self._get_test_tree_ensemble_classifier_binary(post)
            if post in ('NONE',):
                self._check_ort(onx, {'X': x})
            sess = ReferenceEvaluator(onx)
            got = sess.run(None, {'X': x})
            assert_allclose(expected[1], got[1], atol=1e-06)
            assert_allclose(expected[0], got[0])