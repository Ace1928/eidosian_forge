import collections
import functools
import re
import string
import numpy as np
import opt_einsum
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('math.special.dawsn')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def dawsn(x, name=None):
    """Computes Dawson's integral of `x` element-wise.

  Dawson's integral is defined as `exp(-x**2)` times the integral of
  `exp(t**2)` from `0` to `x`, with the domain of definition all real numbers.

  Dawson's function is odd.
  >>> tf.math.special.dawsn([-1., -0.5, 0.5, 1.]).numpy()
  array([-0.5380795, -0.4244364, 0.4244364,  0.5380795], dtype=float32)

  This implementation is based off of the Cephes math library.

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types:
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.dawsn
  @end_compatibility
  """
    with ops.name_scope(name, 'dawsn', [x]):
        return gen_special_math_ops.dawsn(x)