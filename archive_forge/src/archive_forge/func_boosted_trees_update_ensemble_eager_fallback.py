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
def boosted_trees_update_ensemble_eager_fallback(tree_ensemble_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], feature_ids: _atypes.TensorFuzzingAnnotation[_atypes.Int32], node_ids: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], gains: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], thresholds: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], left_node_contribs: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], right_node_contribs: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], max_depth: _atypes.TensorFuzzingAnnotation[_atypes.Int32], learning_rate: _atypes.TensorFuzzingAnnotation[_atypes.Float32], pruning_mode: int, name, ctx):
    if not isinstance(node_ids, (list, tuple)):
        raise TypeError("Expected list for 'node_ids' argument to 'boosted_trees_update_ensemble' Op, not %r." % node_ids)
    _attr_num_features = len(node_ids)
    if not isinstance(gains, (list, tuple)):
        raise TypeError("Expected list for 'gains' argument to 'boosted_trees_update_ensemble' Op, not %r." % gains)
    if len(gains) != _attr_num_features:
        raise ValueError("List argument 'gains' to 'boosted_trees_update_ensemble' Op with length %d must match length %d of argument 'node_ids'." % (len(gains), _attr_num_features))
    if not isinstance(thresholds, (list, tuple)):
        raise TypeError("Expected list for 'thresholds' argument to 'boosted_trees_update_ensemble' Op, not %r." % thresholds)
    if len(thresholds) != _attr_num_features:
        raise ValueError("List argument 'thresholds' to 'boosted_trees_update_ensemble' Op with length %d must match length %d of argument 'node_ids'." % (len(thresholds), _attr_num_features))
    if not isinstance(left_node_contribs, (list, tuple)):
        raise TypeError("Expected list for 'left_node_contribs' argument to 'boosted_trees_update_ensemble' Op, not %r." % left_node_contribs)
    if len(left_node_contribs) != _attr_num_features:
        raise ValueError("List argument 'left_node_contribs' to 'boosted_trees_update_ensemble' Op with length %d must match length %d of argument 'node_ids'." % (len(left_node_contribs), _attr_num_features))
    if not isinstance(right_node_contribs, (list, tuple)):
        raise TypeError("Expected list for 'right_node_contribs' argument to 'boosted_trees_update_ensemble' Op, not %r." % right_node_contribs)
    if len(right_node_contribs) != _attr_num_features:
        raise ValueError("List argument 'right_node_contribs' to 'boosted_trees_update_ensemble' Op with length %d must match length %d of argument 'node_ids'." % (len(right_node_contribs), _attr_num_features))
    pruning_mode = _execute.make_int(pruning_mode, 'pruning_mode')
    tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
    feature_ids = _ops.convert_to_tensor(feature_ids, _dtypes.int32)
    node_ids = _ops.convert_n_to_tensor(node_ids, _dtypes.int32)
    gains = _ops.convert_n_to_tensor(gains, _dtypes.float32)
    thresholds = _ops.convert_n_to_tensor(thresholds, _dtypes.int32)
    left_node_contribs = _ops.convert_n_to_tensor(left_node_contribs, _dtypes.float32)
    right_node_contribs = _ops.convert_n_to_tensor(right_node_contribs, _dtypes.float32)
    max_depth = _ops.convert_to_tensor(max_depth, _dtypes.int32)
    learning_rate = _ops.convert_to_tensor(learning_rate, _dtypes.float32)
    _inputs_flat = [tree_ensemble_handle, feature_ids] + list(node_ids) + list(gains) + list(thresholds) + list(left_node_contribs) + list(right_node_contribs) + [max_depth, learning_rate]
    _attrs = ('pruning_mode', pruning_mode, 'num_features', _attr_num_features)
    _result = _execute.execute(b'BoostedTreesUpdateEnsemble', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result