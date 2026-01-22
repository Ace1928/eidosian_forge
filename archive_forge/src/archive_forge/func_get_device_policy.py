from typing import Union
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.util import _pywrap_determinism
from tensorflow.python.util import _pywrap_tensor_float_32_execution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('config.experimental.get_device_policy')
def get_device_policy():
    """Gets the current device policy.

  The device policy controls how operations requiring inputs on a specific
  device (e.g., on GPU:0) handle inputs on a different device (e.g. GPU:1).

  This function only gets the device policy for the current thread. Any
  subsequently started thread will again use the default policy.

  Returns:
    Current thread device policy
  """
    device_policy = context.context().device_policy
    if device_policy == context.DEVICE_PLACEMENT_SILENT:
        return 'silent'
    elif device_policy == context.DEVICE_PLACEMENT_SILENT_FOR_INT32:
        return 'silent_for_int32'
    elif device_policy == context.DEVICE_PLACEMENT_WARN:
        return 'warn'
    elif device_policy == context.DEVICE_PLACEMENT_EXPLICIT:
        return 'explicit'
    else:
        raise errors.InternalError(f'Got an invalid device policy: {device_policy!r}.')