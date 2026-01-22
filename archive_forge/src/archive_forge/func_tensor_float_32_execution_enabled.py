from typing import Union
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.util import _pywrap_determinism
from tensorflow.python.util import _pywrap_tensor_float_32_execution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('config.experimental.tensor_float_32_execution_enabled')
def tensor_float_32_execution_enabled():
    """Returns whether TensorFloat-32 is enabled.

  By default, TensorFloat-32 is enabled, but this can be changed with
  `tf.config.experimental.enable_tensor_float_32_execution`.

  Returns:
    True if TensorFloat-32 is enabled (the default) and False otherwise
  """
    return _pywrap_tensor_float_32_execution.is_enabled()