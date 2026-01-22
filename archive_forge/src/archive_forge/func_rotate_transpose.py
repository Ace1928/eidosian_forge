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
def rotate_transpose(x, shift, name='rotate_transpose'):
    """Circularly moves dims left or right.

  Effectively identical to:

  ```python
  numpy.transpose(x, numpy.roll(numpy.arange(len(x.shape)), shift))
  ```

  When `validate_args=False` additional graph-runtime checks are
  performed. These checks entail moving data from to GPU to CPU.

  Example:

  ```python
  x = tf.random.normal([1, 2, 3, 4])  # Tensor of shape [1, 2, 3, 4].
  rotate_transpose(x, -1).shape == [2, 3, 4, 1]
  rotate_transpose(x, -2).shape == [3, 4, 1, 2]
  rotate_transpose(x,  1).shape == [4, 1, 2, 3]
  rotate_transpose(x,  2).shape == [3, 4, 1, 2]
  rotate_transpose(x,  7).shape == rotate_transpose(x, 3).shape  # [2, 3, 4, 1]
  rotate_transpose(x, -7).shape == rotate_transpose(x, -3).shape  # [4, 1, 2, 3]
  ```

  Args:
    x: `Tensor`.
    shift: `Tensor`. Number of dimensions to transpose left (shift<0) or
      transpose right (shift>0).
    name: Python `str`. The name to give this op.

  Returns:
    rotated_x: Input `Tensor` with dimensions circularly rotated by shift.

  Raises:
    TypeError: if shift is not integer type.
  """
    with ops.name_scope(name, values=[x, shift]):
        x = ops.convert_to_tensor(x, name='x')
        shift = ops.convert_to_tensor(shift, name='shift')
        check_ops.assert_integer(shift)
        shift_value_static = tensor_util.constant_value(shift)
        ndims = x.get_shape().ndims
        if ndims is not None and shift_value_static is not None:
            if ndims < 2:
                return x
            shift_value_static = np.sign(shift_value_static) * (abs(shift_value_static) % ndims)
            if shift_value_static == 0:
                return x
            perm = np.roll(np.arange(ndims), shift_value_static)
            return array_ops.transpose(x, perm=perm)
        else:
            ndims = array_ops.rank(x)
            shift = array_ops.where_v2(math_ops.less(shift, 0), math_ops.mod(-shift, ndims), ndims - math_ops.mod(shift, ndims))
            first = math_ops.range(0, shift)
            last = math_ops.range(shift, ndims)
            perm = array_ops.concat([last, first], 0)
            return array_ops.transpose(x, perm=perm)