from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['losses.hinge_loss'])
@dispatch.add_dispatch_support
def hinge_loss(labels, logits, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Adds a hinge loss to the training procedure.

  Args:
    labels: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0. Internally
      the {0,1} labels are converted to {-1,1} when calculating the hinge loss.
    logits: The logits, a float tensor. Note that logits are assumed to be
      unbounded and 0-centered. A value > 0 (resp. < 0) is considered a positive
      (resp. negative) binary prediction.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shapes of `logits` and `labels` don't match or
      if `labels` or `logits` is None.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if logits is None:
        raise ValueError('Argument `logits` must not be None.')
    with ops.name_scope(scope, 'hinge_loss', (logits, labels, weights)) as scope:
        logits = math_ops.cast(logits, dtype=dtypes.float32)
        labels = math_ops.cast(labels, dtype=dtypes.float32)
        logits.get_shape().assert_is_compatible_with(labels.get_shape())
        all_ones = array_ops.ones_like(labels)
        labels = math_ops.subtract(2 * labels, all_ones)
        losses = nn_ops.relu(math_ops.subtract(all_ones, math_ops.multiply(labels, logits)))
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)