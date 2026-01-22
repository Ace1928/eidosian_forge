import math
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.platform import device_context
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['nn.weighted_cross_entropy_with_logits'])
@dispatch.add_dispatch_support
@deprecated_args(None, 'targets is deprecated, use labels instead', 'targets')
def weighted_cross_entropy_with_logits(labels=None, logits=None, pos_weight=None, name=None, targets=None):
    """Computes a weighted cross entropy.

  This is like `sigmoid_cross_entropy_with_logits()` except that `pos_weight`,
  allows one to trade off recall and precision by up- or down-weighting the
  cost of a positive error relative to a negative error.

  The usual cross-entropy cost is defined as:

      labels * -log(sigmoid(logits)) +
          (1 - labels) * -log(1 - sigmoid(logits))

  A value `pos_weight > 1` decreases the false negative count, hence increasing
  the recall.
  Conversely setting `pos_weight < 1` decreases the false positive count and
  increases the precision.
  This can be seen from the fact that `pos_weight` is introduced as a
  multiplicative coefficient for the positive labels term
  in the loss expression:

      labels * -log(sigmoid(logits)) * pos_weight +
          (1 - labels) * -log(1 - sigmoid(logits))

  For brevity, let `x = logits`, `z = labels`, `q = pos_weight`.
  The loss is:

        qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
      = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))

  Setting `l = (1 + (q - 1) * z)`, to ensure stability and avoid overflow,
  the implementation uses

      (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

  `logits` and `labels` must have the same type and shape.

  Args:
    labels: A `Tensor` of the same type and shape as `logits`.
    logits: A `Tensor` of type `float32` or `float64`.
    pos_weight: A coefficient to use on the positive examples.
    name: A name for the operation (optional).
    targets: Deprecated alias for labels.

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    weighted logistic losses.

  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
    labels = deprecated_argument_lookup('labels', labels, 'targets', targets)
    return weighted_cross_entropy_with_logits_v2(labels, logits, pos_weight, name)