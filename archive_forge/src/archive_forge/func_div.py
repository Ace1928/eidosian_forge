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
@tf_export(v1=['div'])
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions='Deprecated in favor of operator or tf.math.divide.')
def div(x, y, name=None):
    """Divides x / y elementwise (using Python 2 division operator semantics).

  @compatibility(TF2)
  This function is deprecated in TF2. Prefer using the Tensor division operator,
  `tf.divide`, or `tf.math.divide`, which obey the Python 3 division operator
  semantics.
  @end_compatibility


  This function divides `x` and `y`, forcing Python 2 semantics. That is, if `x`
  and `y` are both integers then the result will be an integer. This is in
  contrast to Python 3, where division with `/` is always a float while division
  with `//` is always an integer.

  Args:
    x: `Tensor` numerator of real numeric type.
    y: `Tensor` denominator of real numeric type.
    name: A name for the operation (optional).

  Returns:
    `x / y` returns the quotient of x and y.
  """
    return _div_python2(x, y, name)