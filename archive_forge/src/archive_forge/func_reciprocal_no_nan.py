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
@tf_export('math.reciprocal_no_nan')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def reciprocal_no_nan(x, name=None):
    """Performs a safe reciprocal operation, element wise.

  If a particular element is zero, the reciprocal for that element is
  also set to zero.

  For example:
  ```python
  x = tf.constant([2.0, 0.5, 0, 1], dtype=tf.float32)
  tf.math.reciprocal_no_nan(x)  # [ 0.5, 2, 0.0, 1.0 ]
  ```

  Args:
    x: A `Tensor` of type `float16`, `float32`, `float64` `complex64` or
      `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as `x`.

  Raises:
    TypeError: x must be of a valid dtype.

  """
    with ops.name_scope(name, 'reciprocal_no_nan', [x]) as scope:
        x = ops.convert_to_tensor(x, name='x')
        one = constant_op.constant(1, dtype=x.dtype.base_dtype, name='one')
        return gen_math_ops.div_no_nan(one, x, name=scope)