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
@tf_export(v1=['math.scalar_mul', 'scalar_mul'])
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def scalar_mul(scalar, x, name=None):
    """Multiplies a scalar times a `Tensor` or `IndexedSlices` object.

  This is a special case of `tf.math.multiply`, where the first value must be a
  `scalar`. Unlike the general form of `tf.math.multiply`, this is operation is
  guaranteed to be efficient for `tf.IndexedSlices`.

  >>> x = tf.reshape(tf.range(30, dtype=tf.float32), [10, 3])
  >>> with tf.GradientTape() as g:
  ...   g.watch(x)
  ...   y = tf.gather(x, [1, 2])  # IndexedSlices
  ...   z = tf.math.scalar_mul(10.0, y)

  Args:
    scalar: A 0-D scalar `Tensor`. Must have known shape.
    x: A `Tensor` or `IndexedSlices` to be scaled.
    name: A name for the operation (optional).

  Returns:
    `scalar * x` of the same type (`Tensor` or `IndexedSlices`) as `x`.

  Raises:
    ValueError: if scalar is not a 0-D `scalar`.
  """
    base_dtype = dtypes.as_dtype(x.dtype).base_dtype
    scalar = ops.convert_to_tensor(scalar, dtype=base_dtype, name='scalar')
    shape = scalar.get_shape()
    if shape.ndims == 0:
        if isinstance(x, indexed_slices.IndexedSlices):
            return indexed_slices.IndexedSlices(gen_math_ops.mul(scalar, x.values, name), x.indices, x.dense_shape)
        else:
            return gen_math_ops.mul(scalar, x, name)
    else:
        raise ValueError(f'The input scalar must be a 0-D value. Received shape {shape}.')