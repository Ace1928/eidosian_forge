from typing import Union
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.util import _pywrap_determinism
from tensorflow.python.util import _pywrap_tensor_float_32_execution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('config.get_soft_device_placement')
def get_soft_device_placement():
    """Return status of soft device placement flag.

  If enabled, ops can be placed on different devices than the device explicitly
  assigned by the user. This potentially has a large performance cost due to an
  increase in data communication between devices.

  Some cases where soft_device_placement would modify device assignment are:
    1. no GPU/TPU implementation for the OP
    2. no GPU devices are known or registered
    3. need to co-locate with reftype input(s) which are from CPU
    4. an OP can not be compiled by XLA.  Common for TPU which always requires
         the XLA compiler.

  For TPUs, if this option is true, a feature called automatic outside
  compilation is enabled. Automatic outside compilation will move uncompilable
  ops within a TPU program to instead run on the host. This can be used when
  encountering compilation failures due to unsupported ops.

  Returns:
   A boolean indicating if soft placement is enabled.
  """
    return context.context().soft_device_placement