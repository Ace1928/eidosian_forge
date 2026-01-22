from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util import dispatch
@dispatch.add_dispatch_support
def exponential(x):
    """Exponential activation function.

  For example:

  >>> a = tf.constant([-3.0,-1.0, 0.0,1.0,3.0], dtype = tf.float32)
  >>> b = tf.keras.activations.exponential(a)
  >>> b.numpy()
  array([0.04978707,  0.36787945,  1.,  2.7182817 , 20.085537], dtype=float32)

  Args:
      x: Input tensor.

  Returns:
      Tensor with exponential activation: `exp(x)`.
  """
    return math_ops.exp(x)