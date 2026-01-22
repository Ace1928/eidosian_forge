import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest
def assert_no_entries_with_modulus_zero(x, message=None, name='assert_no_entries_with_modulus_zero'):
    """Returns `Op` that asserts Tensor `x` has no entries with modulus zero.

  Args:
    x:  Numeric `Tensor`, real, integer, or complex.
    message:  A string message to prepend to failure message.
    name:  A name to give this `Op`.

  Returns:
    An `Op` that asserts `x` has no entries with modulus zero.
  """
    with ops.name_scope(name, values=[x]):
        x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name='x')
        dtype = x.dtype.base_dtype
        should_be_nonzero = math_ops.abs(x)
        zero = tensor_conversion.convert_to_tensor_v2_with_dispatch(0, dtype=dtype.real_dtype)
        return check_ops.assert_less(zero, should_be_nonzero, message=message)