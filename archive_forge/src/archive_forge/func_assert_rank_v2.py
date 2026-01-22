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
@tf_export('debugging.assert_rank', 'assert_rank', v1=[])
@dispatch.add_dispatch_support
def assert_rank_v2(x, rank, message=None, name=None):
    """Assert that `x` has rank equal to `rank`.

  This Op checks that the rank of `x` is equal to `rank`.

  If `x` has a different rank, `message`, as well as the shape of `x` are
  printed, and `InvalidArgumentError` is raised.

  Args:
    x: `Tensor`.
    rank: Scalar integer `Tensor`.
    message: A string to prefix to the default message.
    name: A name for this operation (optional). Defaults to
      "assert_rank".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank.
    If static checks determine `x` has correct rank, a `no_op` is returned.
    This can be used with `tf.control_dependencies` inside of `tf.function`s
    to block followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x` does not have rank `rank`. The check can be performed immediately
      during eager execution or if the shape of `x` is statically known.
  """
    return assert_rank(x=x, rank=rank, message=message, name=name)