from typing import Union
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.util import _pywrap_determinism
from tensorflow.python.util import _pywrap_tensor_float_32_execution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('config.optimizer.set_jit')
@deprecation.deprecated_arg_values(None, '`True` setting is deprecated, use `autoclustering` instead.', warn_once=True, jit_config=True)
def set_optimizer_jit(enabled: Union[bool, str]):
    """Configure JIT compilation.

  Note: compilation is only applied to code that is compiled into a
  graph (in TF2 that's only a code inside `tf.function`).

  Args:
    enabled: JIT compilation configuration.
    Possible values:
     - `"autoclustering"` (`True` is a deprecated alias): perform
     [autoclustering](https://www.tensorflow.org/xla#auto-clustering)
       (automatically identify and compile clusters of nodes) on all graphs
       using
     [XLA](https://www.tensorflow.org/xla).
     - `False`: do not automatically compile any graphs.
  """
    autoclustering_enabled = enabled in (True, 'autoclustering')
    context.context().optimizer_jit = autoclustering_enabled