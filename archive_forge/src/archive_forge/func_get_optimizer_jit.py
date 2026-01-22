from typing import Union
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.util import _pywrap_determinism
from tensorflow.python.util import _pywrap_tensor_float_32_execution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('config.optimizer.get_jit')
def get_optimizer_jit() -> str:
    """Returns JIT compilation configuration for code inside `tf.function`.

  Possible return values:
     -`"autoclustering"` if
     [autoclustering](https://www.tensorflow.org/xla#auto-clustering) is enabled
     - `""` when no default compilation is applied.
  """
    if context.context().optimizer_jit:
        return 'autoclustering'
    return ''