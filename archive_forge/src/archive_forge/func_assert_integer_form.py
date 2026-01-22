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
def assert_integer_form(x, data=None, summarize=None, message=None, int_dtype=None, name='assert_integer_form'):
    """Assert that x has integer components (or floats equal to integers).

  Args:
    x: Floating-point `Tensor`
    data: The tensors to print out if the condition is `False`. Defaults to
      error message and first few entries of `x` and `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    int_dtype: A `tf.dtype` used to cast the float to. The default (`None`)
      implies the smallest possible signed int will be used for casting.
    name: A name for this operation (optional).

  Returns:
    Op raising `InvalidArgumentError` if `cast(x, int_dtype) != x`.
  """
    with ops.name_scope(name, values=[x, data]):
        x = ops.convert_to_tensor(x, name='x')
        if x.dtype.is_integer:
            return control_flow_ops.no_op()
        message = message or '{} has non-integer components'.format(x)
        if int_dtype is None:
            try:
                int_dtype = {dtypes.float16: dtypes.int16, dtypes.float32: dtypes.int32, dtypes.float64: dtypes.int64}[x.dtype.base_dtype]
            except KeyError:
                raise TypeError('Unrecognized type {}'.format(x.dtype.name))
        return check_ops.assert_equal(x, math_ops.cast(math_ops.cast(x, int_dtype), x.dtype), data=data, summarize=summarize, message=message, name=name)