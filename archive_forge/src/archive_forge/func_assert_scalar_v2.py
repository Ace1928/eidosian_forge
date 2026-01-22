import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('debugging.assert_scalar', v1=[])
@dispatch.add_dispatch_support
def assert_scalar_v2(tensor, message=None, name=None):
    """Asserts that the given `tensor` is a scalar.

  This function raises `ValueError` unless it can be certain that the given
  `tensor` is a scalar. `ValueError` is also raised if the shape of `tensor` is
  unknown.

  This is always checked statically, so this method returns nothing.

  Args:
    tensor: A `Tensor`.
    message: A string to prefix to the default message.
    name:  A name for this operation. Defaults to "assert_scalar"

  Raises:
    ValueError: If the tensor is not scalar (rank 0), or if its shape is
      unknown.
  """
    assert_scalar(tensor=tensor, message=message, name=name)