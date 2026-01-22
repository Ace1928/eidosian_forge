import functools
import hashlib
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util import tf_inspect
def softplus_inverse(x, name=None):
    """Computes the inverse softplus, i.e., x = softplus_inverse(softplus(x)).

  Mathematically this op is equivalent to:

  ```none
  softplus_inverse = log(exp(x) - 1.)
  ```

  Args:
    x: `Tensor`. Non-negative (not enforced), floating-point.
    name: A name for the operation (optional).

  Returns:
    `Tensor`. Has the same type/shape as input `x`.
  """
    with ops.name_scope(name, 'softplus_inverse', values=[x]):
        x = ops.convert_to_tensor(x, name='x')
        threshold = np.log(np.finfo(x.dtype.as_numpy_dtype).eps) + 2.0
        is_too_small = math_ops.less(x, np.exp(threshold))
        is_too_large = math_ops.greater(x, -threshold)
        too_small_value = math_ops.log(x)
        too_large_value = x
        x = array_ops.where_v2(math_ops.logical_or(is_too_small, is_too_large), array_ops.ones_like(x), x)
        y = x + math_ops.log(-math_ops.expm1(-x))
        return array_ops.where_v2(is_too_small, too_small_value, array_ops.where_v2(is_too_large, too_large_value, y))