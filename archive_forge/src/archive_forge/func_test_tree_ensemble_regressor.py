from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
@unittest.skipUnless(ONNX_ML, 'ONNX_ML required to test ai.onnx.ml operators')
def test_tree_ensemble_regressor(self) -> None:
    tree = make_node('TreeEnsembleRegressor', ['x'], ['y'], domain=ONNX_ML_DOMAIN, n_targets=5)
    graph = self._make_graph([('x', TensorProto.DOUBLE, (30, 3))], [tree], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (30, 5))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 3), make_opsetid(ONNX_DOMAIN, 11)])