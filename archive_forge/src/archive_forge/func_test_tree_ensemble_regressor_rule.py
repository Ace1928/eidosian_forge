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
@parameterized.expand(tuple(itertools.chain.from_iterable((((Mode.LEQ if opset5 else 'BRANCH_LEQ', np.array([[0.576923], [0.576923], [0.576923]], dtype=np.float32), opset5), (Mode.GT if opset5 else 'BRANCH_GT', np.array([[0.5], [0.5], [0.5]], dtype=np.float32), opset5), (Mode.LT if opset5 else 'BRANCH_LT', np.array([[0.576923], [0.576923], [0.576923]], dtype=np.float32), opset5), (Mode.GTE if opset5 else 'BRANCH_GTE', np.array([[0.5], [0.5], [0.5]], dtype=np.float32), opset5), (Mode.EQ if opset5 else 'BRANCH_EQ', np.array([[1.0], [1.0], [1.0]], dtype=np.float32), opset5), (Mode.NEQ if opset5 else 'BRANCH_NEQ', np.array([[0.076923], [0.076923], [0.076923]], dtype=np.float32), opset5)) for opset5 in [True, False]))))
@unittest.skipIf(not ONNX_ML, reason='onnx not compiled with ai.onnx.ml')
def test_tree_ensemble_regressor_rule(self, rule, expected, opset5):
    x = np.arange(9).reshape((-1, 3)).astype(np.float32) / 10 - 0.5
    model_factory = self._get_test_tree_ensemble_opset_latest if opset5 else self._get_test_tree_ensemble_regressor
    aggregate_function = AggregationFunction.SUM if opset5 else 'SUM'
    model_proto = model_factory(aggregate_function, rule)
    sess = ReferenceEvaluator(model_proto)
    actual, = sess.run(None, {'X': x})
    assert_allclose(expected, actual, atol=1e-06)