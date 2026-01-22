import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def boosted_trees_update_ensemble_v2_eager_fallback(tree_ensemble_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], feature_ids: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], dimension_ids: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], node_ids: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], gains: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], thresholds: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], left_node_contribs: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], right_node_contribs: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], split_types: List[_atypes.TensorFuzzingAnnotation[_atypes.String]], max_depth: _atypes.TensorFuzzingAnnotation[_atypes.Int32], learning_rate: _atypes.TensorFuzzingAnnotation[_atypes.Float32], pruning_mode: _atypes.TensorFuzzingAnnotation[_atypes.Int32], logits_dimension: int, name, ctx):
    if not isinstance(dimension_ids, (list, tuple)):
        raise TypeError("Expected list for 'dimension_ids' argument to 'boosted_trees_update_ensemble_v2' Op, not %r." % dimension_ids)
    _attr_num_features = len(dimension_ids)
    if not isinstance(node_ids, (list, tuple)):
        raise TypeError("Expected list for 'node_ids' argument to 'boosted_trees_update_ensemble_v2' Op, not %r." % node_ids)
    if len(node_ids) != _attr_num_features:
        raise ValueError("List argument 'node_ids' to 'boosted_trees_update_ensemble_v2' Op with length %d must match length %d of argument 'dimension_ids'." % (len(node_ids), _attr_num_features))
    if not isinstance(gains, (list, tuple)):
        raise TypeError("Expected list for 'gains' argument to 'boosted_trees_update_ensemble_v2' Op, not %r." % gains)
    if len(gains) != _attr_num_features:
        raise ValueError("List argument 'gains' to 'boosted_trees_update_ensemble_v2' Op with length %d must match length %d of argument 'dimension_ids'." % (len(gains), _attr_num_features))
    if not isinstance(thresholds, (list, tuple)):
        raise TypeError("Expected list for 'thresholds' argument to 'boosted_trees_update_ensemble_v2' Op, not %r." % thresholds)
    if len(thresholds) != _attr_num_features:
        raise ValueError("List argument 'thresholds' to 'boosted_trees_update_ensemble_v2' Op with length %d must match length %d of argument 'dimension_ids'." % (len(thresholds), _attr_num_features))
    if not isinstance(left_node_contribs, (list, tuple)):
        raise TypeError("Expected list for 'left_node_contribs' argument to 'boosted_trees_update_ensemble_v2' Op, not %r." % left_node_contribs)
    if len(left_node_contribs) != _attr_num_features:
        raise ValueError("List argument 'left_node_contribs' to 'boosted_trees_update_ensemble_v2' Op with length %d must match length %d of argument 'dimension_ids'." % (len(left_node_contribs), _attr_num_features))
    if not isinstance(right_node_contribs, (list, tuple)):
        raise TypeError("Expected list for 'right_node_contribs' argument to 'boosted_trees_update_ensemble_v2' Op, not %r." % right_node_contribs)
    if len(right_node_contribs) != _attr_num_features:
        raise ValueError("List argument 'right_node_contribs' to 'boosted_trees_update_ensemble_v2' Op with length %d must match length %d of argument 'dimension_ids'." % (len(right_node_contribs), _attr_num_features))
    if not isinstance(split_types, (list, tuple)):
        raise TypeError("Expected list for 'split_types' argument to 'boosted_trees_update_ensemble_v2' Op, not %r." % split_types)
    if len(split_types) != _attr_num_features:
        raise ValueError("List argument 'split_types' to 'boosted_trees_update_ensemble_v2' Op with length %d must match length %d of argument 'dimension_ids'." % (len(split_types), _attr_num_features))
    if not isinstance(feature_ids, (list, tuple)):
        raise TypeError("Expected list for 'feature_ids' argument to 'boosted_trees_update_ensemble_v2' Op, not %r." % feature_ids)
    _attr_num_groups = len(feature_ids)
    if logits_dimension is None:
        logits_dimension = 1
    logits_dimension = _execute.make_int(logits_dimension, 'logits_dimension')
    tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
    feature_ids = _ops.convert_n_to_tensor(feature_ids, _dtypes.int32)
    dimension_ids = _ops.convert_n_to_tensor(dimension_ids, _dtypes.int32)
    node_ids = _ops.convert_n_to_tensor(node_ids, _dtypes.int32)
    gains = _ops.convert_n_to_tensor(gains, _dtypes.float32)
    thresholds = _ops.convert_n_to_tensor(thresholds, _dtypes.int32)
    left_node_contribs = _ops.convert_n_to_tensor(left_node_contribs, _dtypes.float32)
    right_node_contribs = _ops.convert_n_to_tensor(right_node_contribs, _dtypes.float32)
    split_types = _ops.convert_n_to_tensor(split_types, _dtypes.string)
    max_depth = _ops.convert_to_tensor(max_depth, _dtypes.int32)
    learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
    pruning_mode = _ops.convert_to_tensor(pruning_mode, _dtypes.int32)
    _inputs_flat = [tree_ensemble_handle] + list(feature_ids) + list(dimension_ids) + list(node_ids) + list(gains) + list(thresholds) + list(left_node_contribs) + list(right_node_contribs) + list(split_types) + [max_depth, learning_rate, pruning_mode]
    _attrs = ('num_features', _attr_num_features, 'logits_dimension', logits_dimension, 'num_groups', _attr_num_groups)
    _result = _execute.execute(b'BoostedTreesUpdateEnsembleV2', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result