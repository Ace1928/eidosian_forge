from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['disable_control_flow_v2'])
def disable_control_flow_v2():
    """Opts out of control flow v2.

  Note: v2 control flow is always enabled inside of tf.function. Calling this
  function has no effect in that case.

  If your code needs tf.disable_control_flow_v2() to be called to work
  properly please file a bug.
  """
    logging.vlog(1, 'Disabling control flow v2')
    ops._control_flow_api_gauge.get_cell().set(False)
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = False