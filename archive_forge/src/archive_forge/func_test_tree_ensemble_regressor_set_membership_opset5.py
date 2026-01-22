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
def test_tree_ensemble_regressor_set_membership_opset5(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    node = make_node('TreeEnsemble', ['X'], ['Y'], domain='ai.onnx.ml', n_targets=4, aggregate_function=AggregationFunction.SUM, membership_values=make_tensor('membership_values', TensorProto.FLOAT, (8,), [1.2, 3.7, 8, 9, np.nan, 12, 7, np.nan]), nodes_missing_value_tracks_true=None, nodes_hitrates=None, post_transform=PostTransform.NONE, tree_roots=[0], nodes_modes=make_tensor('nodes_modes', TensorProto.UINT8, (3,), [Mode.LEQ, Mode.MEMBER, Mode.MEMBER]), nodes_featureids=[0, 0, 0], nodes_splits=make_tensor('nodes_splits', TensorProto.FLOAT, (3,), np.array([11, 232344.0, np.nan], dtype=np.float32)), nodes_trueleafs=[0, 1, 1], nodes_truenodeids=[1, 0, 1], nodes_falseleafs=[1, 0, 1], nodes_falsenodeids=[2, 2, 3], leaf_targetids=[0, 1, 2, 3], leaf_weights=make_tensor('leaf_weights', TensorProto.FLOAT, (4,), [1, 10, 1000, 100]))
    graph = make_graph([node], 'ml', [X], [Y])
    model = make_model_gen_version(graph, opset_imports=OPSETS)
    check_model(model)
    session = ReferenceEvaluator(model)
    X = np.array([1.2, 3.4, -0.12, np.nan, 12, 7], np.float32).reshape(-1, 1)
    expected = np.array([[1, 0, 0, 0], [0, 0, 0, 100], [0, 0, 0, 100], [0, 0, 1000, 0], [0, 0, 1000, 0], [0, 10, 0, 0]], dtype=np.float32)
    output, = session.run(None, {'X': X})
    np.testing.assert_equal(output, expected)