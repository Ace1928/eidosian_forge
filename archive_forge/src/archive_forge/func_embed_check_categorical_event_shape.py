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
def embed_check_categorical_event_shape(categorical_param, name='embed_check_categorical_event_shape'):
    """Embeds checks that categorical distributions don't have too many classes.

  A categorical-type distribution is one which, e.g., returns the class label
  rather than a one-hot encoding.  E.g., `Categorical(probs)`.

  Since distributions output samples in the same dtype as the parameters, we
  must ensure that casting doesn't lose precision. That is, the
  `parameter.dtype` implies a maximum number of classes. However, since shape is
  `int32` and categorical variables are presumed to be indexes into a `Tensor`,
  we must also ensure that the number of classes is no larger than the largest
  possible `int32` index, i.e., `2**31-1`.

  In other words the number of classes, `K`, must satisfy the following
  condition:

  ```python
  K <= min(
      int(2**31 - 1),  # Largest float as an index.
      {
          dtypes.float16: int(2**11),   # Largest int as a float16.
          dtypes.float32: int(2**24),
          dtypes.float64: int(2**53),
      }.get(categorical_param.dtype.base_dtype, 0))
  ```

  Args:
    categorical_param: Floating-point `Tensor` representing parameters of
      distribution over categories. The rightmost shape is presumed to be the
      number of categories.
    name: A name for this operation (optional).

  Returns:
    categorical_param: Input `Tensor` with appropriate assertions embedded.

  Raises:
    TypeError: if `categorical_param` has an unknown `dtype`.
    ValueError: if we can statically identify `categorical_param` as being too
      large (for being closed under int32/float casting).
  """
    with ops.name_scope(name, values=[categorical_param]):
        x = ops.convert_to_tensor(categorical_param, name='categorical_param')
        x_dtype = x.dtype.base_dtype
        max_event_size = _largest_integer_by_dtype(x_dtype) if x_dtype.is_floating else 0
        if max_event_size == 0:
            raise TypeError('Unable to validate size of unrecognized dtype ({}).'.format(x_dtype.name))
        try:
            x_shape_static = x.get_shape().with_rank_at_least(1)
        except ValueError:
            raise ValueError('A categorical-distribution parameter must have at least 1 dimension.')
        if tensor_shape.dimension_value(x_shape_static[-1]) is not None:
            event_size = x_shape_static.dims[-1].value
            if event_size < 2:
                raise ValueError('A categorical-distribution parameter must have at least 2 events.')
            if event_size > max_event_size:
                raise ValueError('Number of classes exceeds `dtype` precision, i.e., {} implies shape ({}) cannot exceed {}.'.format(x_dtype.name, event_size, max_event_size))
            return x
        else:
            event_size = array_ops.shape(x, name='x_shape')[-1]
            return control_flow_ops.with_dependencies([check_ops.assert_rank_at_least(x, 1, message='A categorical-distribution parameter must have at least 1 dimension.'), check_ops.assert_greater_equal(array_ops.shape(x)[-1], 2, message='A categorical-distribution parameter must have at least 2 events.'), check_ops.assert_less_equal(event_size, max_event_size, message='Number of classes exceeds `dtype` precision, i.e., {} dtype cannot exceed {} shape.'.format(x_dtype.name, max_event_size))], x)