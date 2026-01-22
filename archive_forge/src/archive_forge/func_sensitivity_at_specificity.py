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
@tf_export(v1=['metrics.sensitivity_at_specificity'])
def sensitivity_at_specificity(labels, predictions, specificity, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, name=None):
    """Computes the specificity at a given sensitivity.

  The `sensitivity_at_specificity` function creates four local
  variables, `true_positives`, `true_negatives`, `false_positives` and
  `false_negatives` that are used to compute the sensitivity at the given
  specificity value. The threshold for the given specificity value is computed
  and used to evaluate the corresponding sensitivity.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `sensitivity`. `update_op` increments the `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` counts with the weight of each case
  found in the `predictions` and `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  For additional information about specificity and sensitivity, see the
  following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    specificity: A scalar value in range `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    num_thresholds: The number of thresholds to use for matching the given
      specificity.
    metrics_collections: An optional list of collections that `sensitivity`
      should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    sensitivity: A scalar `Tensor` representing the sensitivity at the given
      `specificity` value.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches `sensitivity`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      `specificity` is not between 0 and 1, or if either `metrics_collections`
      or `updates_collections` are not a list or tuple.
    RuntimeError: If eager execution is enabled.
  """
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.sensitivity_at_specificity is not supported when eager execution is enabled.')
    if specificity < 0 or specificity > 1:
        raise ValueError(f'`specificity` must be in the range [0, 1]. Currently, `specificity` got {specificity}.')
    with variable_scope.variable_scope(name, 'sensitivity_at_specificity', (predictions, labels, weights)):
        kepsilon = 1e-07
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]
        values, update_ops = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights)

        def compute_sensitivity_at_specificity(tp, tn, fp, fn, name):
            specificities = math_ops.divide(tn, tn + fp + kepsilon)
            tf_index = math_ops.argmin(math_ops.abs(specificities - specificity), 0)
            tf_index = math_ops.cast(tf_index, dtypes.int32)
            return math_ops.divide(tp[tf_index], tp[tf_index] + fn[tf_index] + kepsilon, name)

        def sensitivity_across_replicas(_, values):
            return compute_sensitivity_at_specificity(values['tp'], values['tn'], values['fp'], values['fn'], 'value')
        sensitivity = _aggregate_across_replicas(metrics_collections, sensitivity_across_replicas, values)
        update_op = compute_sensitivity_at_specificity(update_ops['tp'], update_ops['tn'], update_ops['fp'], update_ops['fn'], 'update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (sensitivity, update_op)