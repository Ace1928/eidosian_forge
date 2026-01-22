import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['placeholder_with_default'])
def placeholder_with_default(input, shape, name=None):
    """A placeholder op that passes through `input` when its output is not fed.

  @compatibility(TF2)
  This API is strongly discouraged for use with eager execution and
  `tf.function`. The primary use of this API is for testing computation wrapped
  within a `tf.function` where the input tensors might not have statically known
  fully-defined shapes. The same can be achieved by creating a
  [concrete function](
  https://www.tensorflow.org/guide/function#obtaining_concrete_functions)
  from the `tf.function` with a `tf.TensorSpec` input which has partially
  defined shapes. For example, the code

  >>> @tf.function
  ... def f():
  ...   x = tf.compat.v1.placeholder_with_default(
  ...       tf.constant([[1., 2., 3.], [4., 5., 6.]]), [None, 3])
  ...   y = tf.constant([[1.],[2.], [3.]])
  ...   z = tf.matmul(x, y)
  ...   assert z.shape[0] == None
  ...   assert z.shape[1] == 1

  >>> f()

  can easily be replaced by

  >>> @tf.function
  ... def f(x):
  ...   y = tf.constant([[1.],[2.], [3.]])
  ...   z = tf.matmul(x, y)
  ...   assert z.shape[0] == None
  ...   assert z.shape[1] == 1

  >>> g = f.get_concrete_function(tf.TensorSpec([None, 3]))

  You can learn more about `tf.function` at [Better
  performance with tf.function](https://www.tensorflow.org/guide/function).
  @end_compatibility

  Args:
    input: A `Tensor`. The default value to produce when output is not fed.
    shape: A `tf.TensorShape` or list of `int`s. The (possibly partial) shape of
      the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    return gen_array_ops.placeholder_with_default(input, shape, name)