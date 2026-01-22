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
@tf_export(v1=['losses.sparse_softmax_cross_entropy'])
@dispatch.add_dispatch_support
def sparse_softmax_cross_entropy(labels, logits, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`.

  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.

  Args:
    labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
      `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
      must be an index in `[0, num_classes)`. Other values will raise an
      exception when this op is run on CPU, and return `NaN` for corresponding
      loss and gradient rows on GPU.
    logits: Unscaled log probabilities of shape
      `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float16`, `float32` or
      `float64`.
    weights: Coefficients for the loss. This must be scalar or broadcastable to
      `labels` (i.e. same rank and each dimension is either 1 or the same).
    scope: the scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has the same shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shapes of `logits`, `labels`, and `weights` are
      incompatible, or if any of them are None.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if logits is None:
        raise ValueError('Argument `logits` must not be None.')
    with ops.name_scope(scope, 'sparse_softmax_cross_entropy_loss', (logits, labels, weights)) as scope:
        labels, logits, weights = _remove_squeezable_dimensions(labels, logits, weights, expected_rank_diff=1)
        losses = nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)