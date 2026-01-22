import builtins
import numbers
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('math.xlog1py')
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def xlog1py(x, y, name=None):
    """Compute x * log1p(y).

  Given `x` and `y`, compute `x * log1p(y)`. This function safely returns
  zero when `x = 0`, no matter what the value of `y` is.

  Example:

  >>> tf.math.xlog1py(0., 1.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.>
  >>> tf.math.xlog1py(1., 1.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.6931472>
  >>> tf.math.xlog1py(2., 2.)
  <tf.Tensor: shape=(), dtype=float32, numpy=2.1972246>
  >>> tf.math.xlog1py(0., -1.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.>

  Args:
    x: A `tf.Tensor` of type `half`, `float32`, `float64`, `complex64`,
      `complex128`
    y: A `tf.Tensor` of type `half`, `float32`, `float64`, `complex64`,
      `complex128`
    name: A name for the operation (optional).

  Returns:
    `x * log1p(y)`.

  @compatibility(scipy)
  Equivalent to scipy.special.xlog1py
  @end_compatibility
  """
    with ops.name_scope(name, 'xlog1py', [x]):
        return gen_math_ops.xlog1py(x, y)