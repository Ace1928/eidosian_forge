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
def boosted_trees_calculate_best_feature_split_v2(node_id_range: _atypes.TensorFuzzingAnnotation[_atypes.Int32], stats_summaries_list: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], split_types: _atypes.TensorFuzzingAnnotation[_atypes.String], candidate_feature_ids: _atypes.TensorFuzzingAnnotation[_atypes.Int32], l1: _atypes.TensorFuzzingAnnotation[_atypes.Float32], l2: _atypes.TensorFuzzingAnnotation[_atypes.Float32], tree_complexity: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_node_weight: _atypes.TensorFuzzingAnnotation[_atypes.Float32], logits_dimension: int, name=None):
    """Calculates gains for each feature and returns the best possible split information for each node. However, if no split is found, then no split information is returned for that node.

  The split information is the best threshold (bucket id), gains and left/right node contributions per node for each feature.

  It is possible that not all nodes can be split on each feature. Hence, the list of possible nodes can differ between the features. Therefore, we return `node_ids_list` for each feature, containing the list of nodes that this feature can be used to split.

  In this manner, the output is the best split per features and per node, so that it needs to be combined later to produce the best split for each node (among all possible features).

  The output shapes are compatible in a way that the first dimension of all tensors are the same and equal to the number of possible split nodes for each feature.

  Args:
    node_id_range: A `Tensor` of type `int32`.
      A Rank 1 tensor (shape=[2]) to specify the range [first, last) of node ids to process within `stats_summary_list`. The nodes are iterated between the two nodes specified by the tensor, as like `for node_id in range(node_id_range[0], node_id_range[1])` (Note that the last index node_id_range[1] is exclusive).
    stats_summaries_list: A list of at least 1 `Tensor` objects with type `float32`.
      A list of Rank 4 tensor (#shape=[max_splits, feature_dims, bucket, stats_dims]) for accumulated stats summary (gradient/hessian) per node, per dimension, per buckets for each feature.
      The first dimension of the tensor is the maximum number of splits, and thus not all elements of it will be used, but only the indexes specified by node_ids will be used.
    split_types: A `Tensor` of type `string`.
      A Rank 1 tensor indicating if this Op should perform inequality split or equality split per feature.
    candidate_feature_ids: A `Tensor` of type `int32`.
      Rank 1 tensor with ids for each feature. This is the real id of the feature.
    l1: A `Tensor` of type `float32`.
      l1 regularization factor on leaf weights, per instance based.
    l2: A `Tensor` of type `float32`.
      l2 regularization factor on leaf weights, per instance based.
    tree_complexity: A `Tensor` of type `float32`.
      adjustment to the gain, per leaf based.
    min_node_weight: A `Tensor` of type `float32`.
      minimum avg of hessians in a node before required for the node to be considered for splitting.
    logits_dimension: An `int` that is `>= 1`.
      The dimension of logit, i.e., number of classes.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (node_ids, gains, feature_ids, feature_dimensions, thresholds, left_node_contribs, right_node_contribs, split_with_default_directions).

    node_ids: A `Tensor` of type `int32`.
    gains: A `Tensor` of type `float32`.
    feature_ids: A `Tensor` of type `int32`.
    feature_dimensions: A `Tensor` of type `int32`.
    thresholds: A `Tensor` of type `int32`.
    left_node_contribs: A `Tensor` of type `float32`.
    right_node_contribs: A `Tensor` of type `float32`.
    split_with_default_directions: A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesCalculateBestFeatureSplitV2', name, node_id_range, stats_summaries_list, split_types, candidate_feature_ids, l1, l2, tree_complexity, min_node_weight, 'logits_dimension', logits_dimension)
            _result = _BoostedTreesCalculateBestFeatureSplitV2Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_calculate_best_feature_split_v2_eager_fallback(node_id_range, stats_summaries_list, split_types, candidate_feature_ids, l1, l2, tree_complexity, min_node_weight, logits_dimension=logits_dimension, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(stats_summaries_list, (list, tuple)):
        raise TypeError("Expected list for 'stats_summaries_list' argument to 'boosted_trees_calculate_best_feature_split_v2' Op, not %r." % stats_summaries_list)
    _attr_num_features = len(stats_summaries_list)
    logits_dimension = _execute.make_int(logits_dimension, 'logits_dimension')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesCalculateBestFeatureSplitV2', node_id_range=node_id_range, stats_summaries_list=stats_summaries_list, split_types=split_types, candidate_feature_ids=candidate_feature_ids, l1=l1, l2=l2, tree_complexity=tree_complexity, min_node_weight=min_node_weight, logits_dimension=logits_dimension, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('num_features', _op._get_attr_int('num_features'), 'logits_dimension', _op._get_attr_int('logits_dimension'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BoostedTreesCalculateBestFeatureSplitV2', _inputs_flat, _attrs, _result)
    _result = _BoostedTreesCalculateBestFeatureSplitV2Output._make(_result)
    return _result