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
def mean_squared_logarithmic_error(y_true, y_pred):
    """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

  `loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.randint(0, 2, size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> y_true = np.maximum(y_true, 1e-7)
  >>> y_pred = np.maximum(y_pred, 1e-7)
  >>> assert np.allclose(
  ...     loss.numpy(),
  ...     np.mean(
  ...         np.square(np.log(y_true + 1.) - np.log(y_pred + 1.)), axis=-1))

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Mean squared logarithmic error values. shape = `[batch_size, d0, .. dN-1]`.
  """
    y_pred = tensor_conversion.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    first_log = math_ops.log(backend.maximum(y_pred, backend.epsilon()) + 1.0)
    second_log = math_ops.log(backend.maximum(y_true, backend.epsilon()) + 1.0)
    return backend.mean(math_ops.squared_difference(first_log, second_log), axis=-1)