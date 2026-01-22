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
def test_scaler(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    node1 = make_node('Scaler', ['X'], ['Y'], scale=[0.5], offset=[-4.5], domain='ai.onnx.ml')
    graph = make_graph([node1], 'ml', [X], [Y])
    onx = make_model_gen_version(graph, opset_imports=OPSETS)
    check_model(onx)
    x = np.arange(12).reshape((3, 4)).astype(np.float32)
    expected = np.array([[2.25, 2.75, 3.25, 3.75], [4.25, 4.75, 5.25, 5.75], [6.25, 6.75, 7.25, 7.75]], dtype=np.float32)
    self._check_ort(onx, {'X': x})
    sess = ReferenceEvaluator(onx)
    got = sess.run(None, {'X': x})[0]
    assert_allclose(expected, got)