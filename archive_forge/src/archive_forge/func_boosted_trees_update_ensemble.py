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
def boosted_trees_update_ensemble(tree_ensemble_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], feature_ids: _atypes.TensorFuzzingAnnotation[_atypes.Int32], node_ids: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], gains: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], thresholds: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], left_node_contribs: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], right_node_contribs: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], max_depth: _atypes.TensorFuzzingAnnotation[_atypes.Int32], learning_rate: _atypes.TensorFuzzingAnnotation[_atypes.Float32], pruning_mode: int, name=None):
    """Updates the tree ensemble by either adding a layer to the last tree being grown

  or by starting a new tree.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
      Handle to the ensemble variable.
    feature_ids: A `Tensor` of type `int32`.
      Rank 1 tensor with ids for each feature. This is the real id of
      the feature that will be used in the split.
    node_ids: A list of `Tensor` objects with type `int32`.
      List of rank 1 tensors representing the nodes for which this feature
      has a split.
    gains: A list with the same length as `node_ids` of `Tensor` objects with type `float32`.
      List of rank 1 tensors representing the gains for each of the feature's
      split.
    thresholds: A list with the same length as `node_ids` of `Tensor` objects with type `int32`.
      List of rank 1 tensors representing the thesholds for each of the
      feature's split.
    left_node_contribs: A list with the same length as `node_ids` of `Tensor` objects with type `float32`.
      List of rank 2 tensors with left leaf contribs for each of
      the feature's splits. Will be added to the previous node values to constitute
      the values of the left nodes.
    right_node_contribs: A list with the same length as `node_ids` of `Tensor` objects with type `float32`.
      List of rank 2 tensors with right leaf contribs for each
      of the feature's splits. Will be added to the previous node values to constitute
      the values of the right nodes.
    max_depth: A `Tensor` of type `int32`. Max depth of the tree to build.
    learning_rate: A `Tensor` of type `float32`.
      shrinkage const for each new tree.
    pruning_mode: An `int` that is `>= 0`.
      0-No pruning, 1-Pre-pruning, 2-Post-pruning.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesUpdateEnsemble', name, tree_ensemble_handle, feature_ids, node_ids, gains, thresholds, left_node_contribs, right_node_contribs, max_depth, learning_rate, 'pruning_mode', pruning_mode)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_update_ensemble_eager_fallback(tree_ensemble_handle, feature_ids, node_ids, gains, thresholds, left_node_contribs, right_node_contribs, max_depth, learning_rate, pruning_mode=pruning_mode, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
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
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesUpdateEnsemble', tree_ensemble_handle=tree_ensemble_handle, feature_ids=feature_ids, node_ids=node_ids, gains=gains, thresholds=thresholds, left_node_contribs=left_node_contribs, right_node_contribs=right_node_contribs, max_depth=max_depth, learning_rate=learning_rate, pruning_mode=pruning_mode, name=name)
    return _op