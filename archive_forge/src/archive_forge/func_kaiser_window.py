import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('signal.kaiser_window')
@dispatch.add_dispatch_support
def kaiser_window(window_length, beta=12.0, dtype=dtypes.float32, name=None):
    """Generate a [Kaiser window][kaiser].

  Args:
    window_length: A scalar `Tensor` indicating the window length to generate.
    beta: Beta parameter for Kaiser window, see reference below.
    dtype: The data type to produce. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.

  [kaiser]:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.kaiser.html
  """
    with ops.name_scope(name, 'kaiser_window'):
        window_length = _check_params(window_length, dtype)
        window_length_const = tensor_util.constant_value(window_length)
        if window_length_const == 1:
            return array_ops.ones([1], dtype=dtype)
        halflen_float = (math_ops.cast(window_length, dtype=dtypes.float32) - 1.0) / 2.0
        arg = math_ops.range(-halflen_float, halflen_float + 0.1, dtype=dtypes.float32)
        arg = math_ops.cast(arg, dtype=dtype)
        beta = math_ops.cast(beta, dtype=dtype)
        one = math_ops.cast(1.0, dtype=dtype)
        halflen_float = math_ops.cast(halflen_float, dtype=dtype)
        num = beta * math_ops.sqrt(nn_ops.relu(one - math_ops.square(arg / halflen_float)))
        window = math_ops.exp(num - beta) * (special_math_ops.bessel_i0e(num) / special_math_ops.bessel_i0e(beta))
    return window