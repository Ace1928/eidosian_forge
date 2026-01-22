from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator
def ragged_and(self, y, name=None):
    """Returns the truth value of elementwise `x & y`.

  Logical AND function.

  Requires that `x` and `y` have the same shape or have
  [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  shapes. For example, `y` can be:

    - A single Python boolean, where the result will be calculated by applying
      logical AND with the single element to each element in `x`.
    - A `tf.Tensor` object of dtype `tf.bool` of the same shape or
      [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
      shape. In this case, the result will be the element-wise logical AND of
      `x` and `y`.
    - A `tf.RaggedTensor` object of dtype `tf.bool` of the same shape or
      [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
      shape. In this case, the result will be the element-wise logical AND of
      `x` and `y`.

  For example:

  >>> # `y` is a Python boolean
  >>> x = tf.ragged.constant([[True, False], [True]])
  >>> y = True
  >>> x & y
  <tf.RaggedTensor [[True, False], [True]]>
  >>> tf.math.logical_and(x, y)  # Equivalent of x & y
  <tf.RaggedTensor [[True, False], [True]]>
  >>> y & x
  <tf.RaggedTensor [[True, False], [True]]>
  >>> tf.math.reduce_all(x & y)  # Reduce to a scalar bool Tensor.
  <tf.Tensor: shape=(), dtype=bool, numpy=False>

  >>> # `y` is a tf.Tensor of the same shape.
  >>> x = tf.ragged.constant([[True, False], [True, False]])
  >>> y = tf.constant([[True, False], [False, True]])
  >>> x & y
  <tf.RaggedTensor [[True, False], [False, False]]>

  >>> # `y` is a tf.Tensor of a broadcast-compatible shape.
  >>> x = tf.ragged.constant([[True, False], [True]])
  >>> y = tf.constant([[True], [False]])
  >>> x & y
  <tf.RaggedTensor [[True, False], [False]]>

  >>> # `y` is a `tf.RaggedTensor` of the same shape.
  >>> x = tf.ragged.constant([[True, False], [True]])
  >>> y = tf.ragged.constant([[False, True], [True]])
  >>> x & y
  <tf.RaggedTensor [[False, False], [True]]>

  >>> # `y` is a `tf.RaggedTensor` of a broadcast-compatible shape.
  >>> x = tf.ragged.constant([[[True, True, False]], [[]], [[True, False]]])
  >>> y = tf.ragged.constant([[[True]], [[True]], [[False]]], ragged_rank=1)
  >>> x & y
  <tf.RaggedTensor [[[True, True, False]], [[]], [[False, False]]]>

  Args:
    y: A Python boolean or a `tf.Tensor` or `tf.RaggedTensor` of dtype
      `tf.bool`.
    name: A name for the operation (optional).

  Returns:
    A `tf.RaggedTensor` of dtype `tf.bool` with the shape that `x` and `y`
    broadcast to.
  """
    return math_ops.logical_and(self, y, name)