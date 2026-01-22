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
def test_tree_ensemble_regressor_missing_opset3(self):
    x = np.arange(9).reshape((-1, 3)).astype(np.float32) / 10 - 0.5
    x[2, 0] = 5
    x[1, :] = np.nan
    expected = np.array([[100001.0], [100100.0], [100100.0]], dtype=np.float32)
    onx = self._get_test_tree_ensemble_regressor('SUM', unique_targets=True)
    self._check_ort(onx, {'X': x}, equal=True)
    sess = ReferenceEvaluator(onx)
    got = sess.run(None, {'X': x})
    assert_allclose(expected, got[0], atol=1e-06)
    self.assertIn('op_type=TreeEnsembleRegressor', str(sess.rt_nodes_[0]))