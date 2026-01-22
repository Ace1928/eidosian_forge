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
@tf_export(v1=['metrics.mean_per_class_accuracy'])
def mean_per_class_accuracy(labels, predictions, num_classes, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """Calculates the mean of the per-class accuracies.

  Calculates the accuracy for each class, then takes the mean of that.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates the accuracy of each class and returns
  them.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since two variables with shape =
      [num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `mean_per_class_accuracy'
      should be added to.
    updates_collections: An optional list of collections `update_op` should be
      added to.
    name: An optional variable_scope name.

  Returns:
    mean_accuracy: A `Tensor` representing the mean per class accuracy.
    update_op: An operation that updates the accuracy tensor.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean_per_class_accuracy is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'mean_accuracy', (predictions, labels, weights)):
        labels = math_ops.cast(labels, dtypes.int64)
        if labels.get_shape().ndims > 1:
            labels = array_ops.reshape(labels, [-1])
        if predictions.get_shape().ndims > 1:
            predictions = array_ops.reshape(predictions, [-1])
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        total = metric_variable([num_classes], dtypes.float32, name='total')
        count = metric_variable([num_classes], dtypes.float32, name='count')
        ones = array_ops.ones([array_ops.size(labels)], dtypes.float32)
        if labels.dtype != predictions.dtype:
            predictions = math_ops.cast(predictions, labels.dtype)
        is_correct = math_ops.cast(math_ops.equal(predictions, labels), dtypes.float32)
        if weights is not None:
            if weights.get_shape().ndims > 1:
                weights = array_ops.reshape(weights, [-1])
            weights = math_ops.cast(weights, dtypes.float32)
            is_correct *= weights
            ones *= weights
        update_total_op = state_ops.scatter_add(total, labels, ones)
        update_count_op = state_ops.scatter_add(count, labels, is_correct)

        def compute_mean_accuracy(_, count, total):
            per_class_accuracy = math_ops.div_no_nan(count, math_ops.maximum(total, 0), name=None)
            mean_accuracy_v = math_ops.reduce_mean(per_class_accuracy, name='mean_accuracy')
            return mean_accuracy_v
        mean_accuracy_v = _aggregate_across_replicas(metrics_collections, compute_mean_accuracy, count, total)
        update_op = math_ops.div_no_nan(update_count_op, math_ops.maximum(update_total_op, 0), name='update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (mean_accuracy_v, update_op)