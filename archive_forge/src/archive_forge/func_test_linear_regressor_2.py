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
def test_linear_regressor_2(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    node1 = make_node('LinearRegressor', ['X'], ['Y'], domain='ai.onnx.ml', coefficients=[0.3, -0.77, 0.3, -0.77], intercepts=[0.5, 0.7], post_transform='NONE', targets=2)
    graph = make_graph([node1], 'ml', [X], [Y])
    onx = make_model_gen_version(graph, opset_imports=OPSETS)
    check_model(onx)
    x = np.arange(6).reshape((-1, 2)).astype(np.float32)
    expected = np.array([[-0.27, -0.07], [-1.21, -1.01], [-2.15, -1.95]], dtype=np.float32)
    self._check_ort(onx, {'X': x}, equal=True)
    sess = ReferenceEvaluator(onx)
    got = sess.run(None, {'X': x})
    assert_allclose(expected, got[0], atol=1e-06)