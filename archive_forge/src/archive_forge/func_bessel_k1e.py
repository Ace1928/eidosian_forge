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
@tf_export('math.special.bessel_k1e')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def bessel_k1e(x, name=None):
    """Computes the Bessel k1e function of `x` element-wise.

  Modified Bessel function of order 1.

  >>> tf.math.special.bessel_k1e([0.5, 1., 2., 4.]).numpy()
  array([2.73100971, 1.63615349, 1.03347685, 0.68157595], dtype=float32)

  Args:
    x: A `Tensor` or `SparseTensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.k1e
  @end_compatibility
  """
    with ops.name_scope(name, 'bessel_k1e', [x]):
        return gen_special_math_ops.bessel_k1e(x)