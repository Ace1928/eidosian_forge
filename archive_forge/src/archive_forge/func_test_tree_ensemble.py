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
@parameterized.expand([TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.FLOAT16])
@unittest.skipUnless(ONNX_ML, 'ONNX_ML required to test ai.onnx.ml operators')
def test_tree_ensemble(self, dtype) -> None:
    interior_nodes = 5
    leaves = 9
    tree = make_node('TreeEnsemble', ['x'], ['y'], domain=ONNX_ML_DOMAIN, n_targets=5, nodes_featureids=[0] * interior_nodes, nodes_splits=make_tensor('nodes_splits', dtype, (interior_nodes,), list(range(interior_nodes))), nodes_modes=make_tensor('nodes_modes', TensorProto.UINT8, (interior_nodes,), [0] * interior_nodes), nodes_truenodeids=[0] * interior_nodes, nodes_falsenodeids=[0] * interior_nodes, nodes_trueleafs=[0] * interior_nodes, nodes_falseleafs=[0] * interior_nodes, membership_values=make_tensor('membership_values', dtype, (7,), [0.0, 0.1, 0.2, np.nan, 0.4, 0.5, 1.0]), leaf_targetids=[0] * leaves, leaf_weights=make_tensor('leaf_weights', dtype, (leaves,), [1] * leaves), tree_roots=[0])
    graph = self._make_graph([('x', dtype, ('Batch Size', 'Features'))], [tree], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', dtype, ('Batch Size', 5))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 5), make_opsetid(ONNX_DOMAIN, 11)])