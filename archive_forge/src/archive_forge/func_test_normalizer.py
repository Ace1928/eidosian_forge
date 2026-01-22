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
def test_normalizer(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    x = np.arange(12).reshape((3, 4)).astype(np.float32)
    expected = {'MAX': x / x.max(axis=1, keepdims=1), 'L1': x / np.abs(x).sum(axis=1, keepdims=1), 'L2': x / (x ** 2).sum(axis=1, keepdims=1) ** 0.5}
    for norm, value in expected.items():
        with self.subTest(norm=norm):
            node1 = make_node('Normalizer', ['X'], ['Y'], norm=norm, domain='ai.onnx.ml')
            graph = make_graph([node1], 'ml', [X], [Y])
            onx = make_model_gen_version(graph, opset_imports=OPSETS)
            check_model(onx)
            feeds = {'X': x}
            self._check_ort(onx, feeds, atol=1e-06)
            sess = ReferenceEvaluator(onx)
            got = sess.run(None, feeds)[0]
            assert_allclose(value, got, atol=1e-06)