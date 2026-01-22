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
@tf_export(v1=['debugging.assert_integer', 'assert_integer'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_integer')
def assert_integer(x, message=None, name=None):
    """Assert that `x` is of integer dtype.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_integer(x)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x: `Tensor` whose basetype is integer and is not quantized.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_integer".

  Raises:
    TypeError:  If `x.dtype` is anything other than non-quantized integer.

  Returns:
    A `no_op` that does nothing.  Type can be determined statically.
  """
    with ops.name_scope(name, 'assert_integer', [x]):
        x = ops.convert_to_tensor(x, name='x')
        if not x.dtype.is_integer:
            if context.executing_eagerly():
                name = 'tensor'
            else:
                name = x.name
            err_msg = '%sExpected "x" to be integer type.  Found: %s of dtype %s' % (_message_prefix(message), name, x.dtype)
            raise TypeError(err_msg)
        return control_flow_ops.no_op('statically_determined_was_integer')