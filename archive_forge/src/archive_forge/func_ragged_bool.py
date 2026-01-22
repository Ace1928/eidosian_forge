from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator
def ragged_bool(self):
    """Raises TypeError when a RaggedTensor is used as a Python bool.

  To prevent RaggedTensor from being used as a bool, this function always raise
  TypeError when being called.

  For example:

  >>> x = tf.ragged.constant([[1, 2], [3]])
  >>> result = True if x else False  # Evaluate x as a bool value.
  Traceback (most recent call last):
  ...
  TypeError: RaggedTensor may not be used as a boolean.

  >>> x = tf.ragged.constant([[1]])
  >>> r = (x == 1)  # tf.RaggedTensor [[True]]
  >>> if r:  # Evaluate r as a bool value.
  ...   pass
  Traceback (most recent call last):
  ...
  TypeError: RaggedTensor may not be used as a boolean.
  """
    raise TypeError('RaggedTensor may not be used as a boolean.')