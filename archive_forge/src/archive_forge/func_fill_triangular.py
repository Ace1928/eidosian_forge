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
def fill_triangular(x, upper=False, name=None):
    """Creates a (batch of) triangular matrix from a vector of inputs.

  Created matrix can be lower- or upper-triangular. (It is more efficient to
  create the matrix as upper or lower, rather than transpose.)

  Triangular matrix elements are filled in a clockwise spiral. See example,
  below.

  If `x.get_shape()` is `[b1, b2, ..., bB, d]` then the output shape is
  `[b1, b2, ..., bB, n, n]` where `n` is such that `d = n(n+1)/2`, i.e.,
  `n = int(np.sqrt(0.25 + 2. * m) - 0.5)`.

  Example:

  ```python
  fill_triangular([1, 2, 3, 4, 5, 6])
  # ==> [[4, 0, 0],
  #      [6, 5, 0],
  #      [3, 2, 1]]

  fill_triangular([1, 2, 3, 4, 5, 6], upper=True)
  # ==> [[1, 2, 3],
  #      [0, 5, 6],
  #      [0, 0, 4]]
  ```

  For comparison, a pure numpy version of this function can be found in
  `util_test.py`, function `_fill_triangular`.

  Args:
    x: `Tensor` representing lower (or upper) triangular elements.
    upper: Python `bool` representing whether output matrix should be upper
      triangular (`True`) or lower triangular (`False`, default).
    name: Python `str`. The name to give this op.

  Returns:
    tril: `Tensor` with lower (or upper) triangular elements filled from `x`.

  Raises:
    ValueError: if `x` cannot be mapped to a triangular matrix.
  """
    with ops.name_scope(name, 'fill_triangular', values=[x]):
        x = ops.convert_to_tensor(x, name='x')
        if tensor_shape.dimension_value(x.shape.with_rank_at_least(1)[-1]) is not None:
            m = np.int32(x.shape.dims[-1].value)
            n = np.sqrt(0.25 + 2.0 * m) - 0.5
            if n != np.floor(n):
                raise ValueError('Input right-most shape ({}) does not correspond to a triangular matrix.'.format(m))
            n = np.int32(n)
            static_final_shape = x.shape[:-1].concatenate([n, n])
        else:
            m = array_ops.shape(x)[-1]
            n = math_ops.cast(math_ops.sqrt(0.25 + math_ops.cast(2 * m, dtype=dtypes.float32)), dtype=dtypes.int32)
            static_final_shape = x.shape.with_rank_at_least(1)[:-1].concatenate([None, None])
        ndims = prefer_static_rank(x)
        if upper:
            x_list = [x, array_ops.reverse(x[..., n:], axis=[ndims - 1])]
        else:
            x_list = [x[..., n:], array_ops.reverse(x, axis=[ndims - 1])]
        new_shape = static_final_shape.as_list() if static_final_shape.is_fully_defined() else array_ops.concat([array_ops.shape(x)[:-1], [n, n]], axis=0)
        x = array_ops.reshape(array_ops.concat(x_list, axis=-1), new_shape)
        x = array_ops.matrix_band_part(x, num_lower=0 if upper else -1, num_upper=-1 if upper else 0)
        x.set_shape(static_final_shape)
        return x