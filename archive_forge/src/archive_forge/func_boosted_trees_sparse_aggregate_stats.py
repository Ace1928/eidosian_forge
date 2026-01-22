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
def boosted_trees_sparse_aggregate_stats(node_ids: _atypes.TensorFuzzingAnnotation[_atypes.Int32], gradients: _atypes.TensorFuzzingAnnotation[_atypes.Float32], hessians: _atypes.TensorFuzzingAnnotation[_atypes.Float32], feature_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int32], feature_values: _atypes.TensorFuzzingAnnotation[_atypes.Int32], feature_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int32], max_splits: int, num_buckets: int, name=None):
    """Aggregates the summary of accumulated stats for the batch.

  The summary stats contains gradients and hessians accumulated for each node, bucket and dimension id.

  Args:
    node_ids: A `Tensor` of type `int32`.
      int32; Rank 1 Tensor containing node ids for each example, shape [batch_size].
    gradients: A `Tensor` of type `float32`.
      float32; Rank 2 Tensor (shape=[batch_size, logits_dimension]) with gradients for each example.
    hessians: A `Tensor` of type `float32`.
      float32; Rank 2 Tensor (shape=[batch_size, hessian_dimension]) with hessians for each example.
    feature_indices: A `Tensor` of type `int32`.
      int32; Rank 2 indices of feature sparse Tensors (shape=[number of sparse entries, 2]).
      Number of sparse entries across all instances from the batch. The first value is
      the index of the instance, the second is dimension of the feature. The second axis
      can only have 2 values, i.e., the input dense version of Tensor can only be matrix.
    feature_values: A `Tensor` of type `int32`.
      int32; Rank 1 values of feature sparse Tensors (shape=[number of sparse entries]).
      Number of sparse entries across all instances from the batch. The first value is
      the index of the instance, the second is dimension of the feature.
    feature_shape: A `Tensor` of type `int32`.
      int32; Rank 1 dense shape of feature sparse Tensors (shape=[2]).
      The first axis can only have 2 values, [batch_size, feature_dimension].
    max_splits: An `int` that is `>= 1`.
      int; the maximum number of splits possible in the whole tree.
    num_buckets: An `int` that is `>= 1`.
      int; equals to the maximum possible value of bucketized feature + 1.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (stats_summary_indices, stats_summary_values, stats_summary_shape).

    stats_summary_indices: A `Tensor` of type `int32`.
    stats_summary_values: A `Tensor` of type `float32`.
    stats_summary_shape: A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesSparseAggregateStats', name, node_ids, gradients, hessians, feature_indices, feature_values, feature_shape, 'max_splits', max_splits, 'num_buckets', num_buckets)
            _result = _BoostedTreesSparseAggregateStatsOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_sparse_aggregate_stats_eager_fallback(node_ids, gradients, hessians, feature_indices, feature_values, feature_shape, max_splits=max_splits, num_buckets=num_buckets, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    max_splits = _execute.make_int(max_splits, 'max_splits')
    num_buckets = _execute.make_int(num_buckets, 'num_buckets')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesSparseAggregateStats', node_ids=node_ids, gradients=gradients, hessians=hessians, feature_indices=feature_indices, feature_values=feature_values, feature_shape=feature_shape, max_splits=max_splits, num_buckets=num_buckets, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('max_splits', _op._get_attr_int('max_splits'), 'num_buckets', _op._get_attr_int('num_buckets'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BoostedTreesSparseAggregateStats', _inputs_flat, _attrs, _result)
    _result = _BoostedTreesSparseAggregateStatsOutput._make(_result)
    return _result