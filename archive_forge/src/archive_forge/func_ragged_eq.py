from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator
def ragged_eq(self, other):
    """Returns result of elementwise `==` or False if not broadcast-compatible.

  Compares two ragged tensors elemewise for equality if they are
  broadcast-compatible; or returns False if they are not
  [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

  Note that this behavior differs from `tf.math.equal`, which raises an
  exception if the two ragged tensors are not broadcast-compatible.

  For example:

  >>> rt1 = tf.ragged.constant([[1, 2], [3]])
  >>> rt1 == rt1
  <tf.RaggedTensor [[True, True], [True]]>

  >>> rt2 = tf.ragged.constant([[1, 2], [4]])
  >>> rt1 == rt2
  <tf.RaggedTensor [[True, True], [False]]>

  >>> rt3 = tf.ragged.constant([[1, 2], [3, 4]])
  >>> # rt1 and rt3 are not broadcast-compatible.
  >>> rt1 == rt3
  False

  >>> # You can also compare a `tf.RaggedTensor` to a `tf.Tensor`.
  >>> t = tf.constant([[1, 2], [3, 4]])
  >>> rt1 == t
  False
  >>> t == rt1
  False
  >>> rt4 = tf.ragged.constant([[1, 2], [3, 4]])
  >>> rt4 == t
  <tf.RaggedTensor [[True, True], [True, True]]>
  >>> t == rt4
  <tf.RaggedTensor [[True, True], [True, True]]>

  Args:
    other: The right-hand side of the `==` operator.

  Returns:
    The ragged tensor result of the elementwise `==` operation, or `False` if
    the arguments are not broadcast-compatible.
  """
    return math_ops.tensor_equals(self, other)