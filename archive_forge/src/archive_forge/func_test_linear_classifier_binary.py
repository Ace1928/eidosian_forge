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
def test_linear_classifier_binary(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    In = make_tensor_value_info('I', TensorProto.INT64, [None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    expected_post = {'NONE': [np.array([1, 1], dtype=np.int64), np.array([[-9.53, 9.53], [-6.65, 6.65]], dtype=np.float32)], 'LOGISTIC': [np.array([1, 1], dtype=np.int64), np.array([[7.263436e-05, 0.9999274], [0.00129235, 0.9987077]], dtype=np.float32)], 'SOFTMAX': [np.array([1, 1], dtype=np.int64), np.array([[5.276517e-09, 1.0], [1.674492e-06, 0.9999983]], dtype=np.float32)], 'SOFTMAX_ZERO': [np.array([1, 1], dtype=np.int64), np.array([[5.276517e-09, 1.0], [1.674492e-06, 0.9999983]], dtype=np.float32)]}
    x = np.arange(6).reshape((-1, 3)).astype(np.float32)
    for post in ('SOFTMAX', 'NONE', 'LOGISTIC', 'SOFTMAX_ZERO'):
        expected = expected_post[post]
        with self.subTest(post_transform=post):
            node1 = make_node('LinearClassifier', ['X'], ['I', 'Y'], domain='ai.onnx.ml', classlabels_ints=[0, 1], coefficients=[-0.58, -0.29, -0.09], intercepts=[10.0], multi_class=0, post_transform=post)
            graph = make_graph([node1], 'ml', [X], [In, Y])
            onx = make_model_gen_version(graph, opset_imports=OPSETS)
            check_model(onx)
            sess = ReferenceEvaluator(onx)
            got = sess.run(None, {'X': x})
            assert_allclose(expected[1], got[1], atol=1e-06)
            assert_allclose(expected[0], got[0])