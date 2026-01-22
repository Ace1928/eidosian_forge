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
def test_one_hot_encoder_int(self):
    X = make_tensor_value_info('X', TensorProto.INT64, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None])
    node1 = make_node('OneHotEncoder', ['X'], ['Y'], domain='ai.onnx.ml', zeros=1, cats_int64s=[1, 2, 3])
    graph = make_graph([node1], 'ml', [X], [Y])
    onx = make_model_gen_version(graph, opset_imports=OPSETS)
    check_model(onx)
    x = np.array([[5, 1, 3], [2, 1, 3]], dtype=np.int64)
    expected = np.array([[[0, 0, 0], [1, 0, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0], [0, 0, 1]]], dtype=np.float32)
    self._check_ort(onx, {'X': x}, equal=True)
    sess = ReferenceEvaluator(onx)
    got = sess.run(None, {'X': x})[0]
    self.assertEqual(expected.tolist(), got.tolist())