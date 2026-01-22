from typing import Text
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['tpu.core'])
def core(num: int) -> Text:
    """Returns the device name for a core in a replicated TPU computation.

  Args:
    num: the virtual core number within each replica to which operators should
    be assigned.
  Returns:
    A device name, suitable for passing to `tf.device()`.
  """
    return 'device:TPU_REPLICATED_CORE:{}'.format(num)