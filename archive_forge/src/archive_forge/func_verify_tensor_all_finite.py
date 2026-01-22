from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['debugging.assert_all_finite', 'verify_tensor_all_finite'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('verify_tensor_all_finite')
def verify_tensor_all_finite(t=None, msg=None, name=None, x=None, message=None):
    """Assert that the tensor does not contain any NaN's or Inf's.

  Args:
    t: Tensor to check.
    msg: Message to log on failure.
    name: A name for this operation (optional).
    x: Alias for t.
    message: Alias for msg.

  Returns:
    Same tensor as `t`.
  """
    x = deprecation.deprecated_argument_lookup('x', x, 't', t)
    message = deprecation.deprecated_argument_lookup('message', message, 'msg', msg)
    return verify_tensor_all_finite_v2(x, message, name)