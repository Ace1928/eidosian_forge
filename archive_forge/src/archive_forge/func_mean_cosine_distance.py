from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['metrics.mean_cosine_distance'])
def mean_cosine_distance(labels, predictions, dim, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """Computes the cosine distance between the labels and predictions.

  The `mean_cosine_distance` function creates two local variables,
  `total` and `count` that are used to compute the average cosine distance
  between `predictions` and `labels`. This average is weighted by `weights`,
  and it is ultimately returned as `mean_distance`, which is an idempotent
  operation that simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `mean_distance`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` of arbitrary shape.
    predictions: A `Tensor` of the same shape as `labels`.
    dim: The dimension along which the cosine distance is computed.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension). Also,
      dimension `dim` must be `1`.
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    mean_distance: A `Tensor` representing the current mean, the value of
      `total` divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean_cosine_distance is not supported when eager execution is enabled.')
    predictions, labels, weights = _remove_squeezable_dimensions(predictions=predictions, labels=labels, weights=weights)
    radial_diffs = math_ops.multiply(predictions, labels)
    radial_diffs = math_ops.reduce_sum(radial_diffs, axis=[dim], keepdims=True)
    mean_distance, update_op = mean(radial_diffs, weights, None, None, name or 'mean_cosine_distance')
    mean_distance = math_ops.subtract(1.0, mean_distance)
    update_op = math_ops.subtract(1.0, update_op)
    if metrics_collections:
        ops.add_to_collections(metrics_collections, mean_distance)
    if updates_collections:
        ops.add_to_collections(updates_collections, update_op)
    return (mean_distance, update_op)