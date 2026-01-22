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
@tf_export(v1=['debugging.assert_rank_at_least', 'assert_rank_at_least'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_rank_at_least')
def assert_rank_at_least(x, rank, data=None, summarize=None, message=None, name=None):
    """Assert `x` has rank equal to `rank` or higher.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_rank_at_least(x, 2)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_rank_at_least".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank or higher.
    If static checks determine `x` has correct rank, a `no_op` is returned.

  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  """
    with ops.name_scope(name, 'assert_rank_at_least', (x, rank) + tuple(data or [])):
        x = ops.convert_to_tensor(x, name='x')
        rank = ops.convert_to_tensor(rank, name='rank')
        message = _message_prefix(message)
        static_condition = lambda actual_rank, given_rank: actual_rank >= given_rank
        dynamic_condition = math_ops.greater_equal
        if context.executing_eagerly():
            name = ''
        else:
            name = x.name
        if data is None:
            data = [message, 'Tensor %s must have rank at least' % name, rank, 'Received shape: ', array_ops.shape(x)]
        try:
            assert_op = _assert_rank_condition(x, rank, static_condition, dynamic_condition, data, summarize)
        except ValueError as e:
            if e.args[0] == 'Static rank condition failed':
                raise ValueError('%sTensor %s must have rank at least %d.  Received rank %d, shape %s' % (message, name, e.args[2], e.args[1], x.get_shape()))
            else:
                raise
    return assert_op