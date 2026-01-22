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
@parameterized.expand([{'nodes_truenodeids': [0] * 6, 'leaf_weights': make_tensor('leaf_weights', TensorProto.DOUBLE, (9,), [1] * 9), 'nodes_splits': make_tensor('nodes_splits', TensorProto.DOUBLE, (5,), [1] * 5)}, {'nodes_truenodeids': [0] * 5, 'leaf_weights': make_tensor('leaf_weights', TensorProto.FLOAT, (9,), [1] * 9), 'nodes_splits': make_tensor('nodes_splits', TensorProto.DOUBLE, (5,), [1] * 5)}, {'nodes_truenodeids': [0] * 5, 'leaf_weights': make_tensor('leaf_weights', TensorProto.DOUBLE, (18,), [1] * 18), 'nodes_splits': make_tensor('nodes_splits', TensorProto.DOUBLE, (5,), [1] * 5)}, {'nodes_truenodeids': [0] * 5, 'leaf_weights': make_tensor('leaf_weights', TensorProto.DOUBLE, (9,), [1] * 9), 'nodes_splits': make_tensor('nodes_splits', TensorProto.FLOAT, (5,), [1] * 5)}])
@unittest.skipUnless(ONNX_ML, 'ONNX_ML required to test ai.onnx.ml operators')
def test_tree_ensemble_fails_if_invalid_attributes(self, nodes_truenodeids, leaf_weights, nodes_splits) -> None:
    interior_nodes = 5
    leaves = 9
    tree = make_node('TreeEnsemble', ['x'], ['y'], domain=ONNX_ML_DOMAIN, n_targets=5, nodes_featureids=[0] * interior_nodes, nodes_splits=nodes_splits, nodes_modes=make_tensor('nodes_modes', TensorProto.UINT8, (interior_nodes,), [0] * interior_nodes), nodes_truenodeids=nodes_truenodeids, nodes_falsenodeids=[0] * interior_nodes, nodes_trueleafs=[0] * interior_nodes, nodes_falseleafs=[0] * interior_nodes, leaf_targetids=[0] * leaves, leaf_weights=leaf_weights, tree_roots=[0])
    graph = self._make_graph([('x', TensorProto.DOUBLE, ('Batch Size', 'Features'))], [tree], [])
    self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)