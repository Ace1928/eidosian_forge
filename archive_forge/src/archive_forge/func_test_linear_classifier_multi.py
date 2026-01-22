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
def test_linear_classifier_multi(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    In = make_tensor_value_info('I', TensorProto.INT64, [None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    expected_post = {'NONE': [np.array([0, 2, 2], dtype=np.int64), np.array([[2.41, -2.12, 0.59], [0.67, -1.14, 1.35], [-1.07, -0.16, 2.11]], dtype=np.float32)], 'LOGISTIC': [np.array([0, 2, 2], dtype=np.int64), np.array([[0.917587, 0.107168, 0.643365], [0.661503, 0.24232, 0.79413], [0.255403, 0.460085, 0.891871]], dtype=np.float32)], 'SOFTMAX': [np.array([0, 2, 2], dtype=np.int64), np.array([[0.852656, 0.009192, 0.138152], [0.318722, 0.05216, 0.629118], [0.036323, 0.090237, 0.87344]], dtype=np.float32)], 'SOFTMAX_ZERO': [np.array([0, 2, 2], dtype=np.int64), np.array([[0.852656, 0.009192, 0.138152], [0.318722, 0.05216, 0.629118], [0.036323, 0.090237, 0.87344]], dtype=np.float32)], 'PROBIT': [np.array([1, 1, 1], dtype=np.int64), np.array([[-0.527324, -0.445471, -1.080504], [-0.067731, 0.316014, -0.310748], [0.377252, 1.405167, 0.295001]], dtype=np.float32)]}
    for post in ('SOFTMAX', 'NONE', 'LOGISTIC', 'SOFTMAX_ZERO', 'PROBIT'):
        if post == 'PROBIT':
            coefficients = [0.058, 0.029, 0.09, 0.058, 0.029, 0.09]
            intercepts = [0.27, 0.27, 0.05]
        else:
            coefficients = [-0.58, -0.29, -0.09, 0.58, 0.29, 0.09]
            intercepts = [2.7, -2.7, 0.5]
        with self.subTest(post_transform=post):
            node1 = make_node('LinearClassifier', ['X'], ['I', 'Y'], domain='ai.onnx.ml', classlabels_ints=[0, 1, 2], coefficients=coefficients, intercepts=intercepts, multi_class=0, post_transform=post)
            graph = make_graph([node1], 'ml', [X], [In, Y])
            onx = make_model_gen_version(graph, opset_imports=OPSETS)
            check_model(onx)
            x = np.arange(6).reshape((-1, 2)).astype(np.float32)
            self._check_ort(onx, {'X': x}, rev=True, atol=0.0001)
            sess = ReferenceEvaluator(onx)
            got = sess.run(None, {'X': x})
            expected = expected_post[post]
            assert_allclose(expected[1], got[1], atol=0.0001)
            assert_allclose(expected[0], got[0])