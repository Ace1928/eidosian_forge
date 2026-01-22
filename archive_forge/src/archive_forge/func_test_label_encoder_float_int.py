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
def test_label_encoder_float_int(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.INT64, [None, None])
    node1 = make_node('LabelEncoder', ['X'], ['Y'], domain='ai.onnx.ml', default_int64=-5, keys_floats=[4.0, 1.0, 2.0, 3.0], values_int64s=[0, 1, 2, 3])
    graph = make_graph([node1], 'ml', [X], [Y])
    onx = make_model_gen_version(graph, opset_imports=OPSETS)
    check_model(onx)
    x = np.array([[0, 1, np.nan, 3, 4]], dtype=np.float32).T
    expected = np.array([[-5, 1, -5, 3, 0]], dtype=np.int64).T
    self._check_ort(onx, {'X': x})
    sess = ReferenceEvaluator(onx)
    got = sess.run(None, {'X': x})[0]
    assert_allclose(expected, got)