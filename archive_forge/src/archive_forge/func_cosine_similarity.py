import abc
import functools
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.tools.docs import doc_controls
@dispatch.add_dispatch_support
def cosine_similarity(y_true, y_pred, axis=-1):
    """Computes the cosine similarity between labels and predictions.

  Note that it is a number between -1 and 1. When it is a negative number
  between -1 and 0, 0 indicates orthogonality and values closer to -1
  indicate greater similarity. The values closer to 1 indicate greater
  dissimilarity. This makes it usable as a loss function in a setting
  where you try to maximize the proximity between predictions and
  targets. If either `y_true` or `y_pred` is a zero vector, cosine
  similarity will be 0 regardless of the proximity between predictions
  and targets.

  `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`

  Standalone usage:

  >>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
  >>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
  >>> loss = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=1)
  >>> loss.numpy()
  array([-0., -0.999, 0.999], dtype=float32)

  Args:
    y_true: Tensor of true targets.
    y_pred: Tensor of predicted targets.
    axis: Axis along which to determine similarity.

  Returns:
    Cosine similarity tensor.
  """
    y_true = nn.l2_normalize(y_true, axis=axis)
    y_pred = nn.l2_normalize(y_pred, axis=axis)
    return -math_ops.reduce_sum(y_true * y_pred, axis=axis)