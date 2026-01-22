import atexit
import os
import re
import socket
import threading
import uuid
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import function as function_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
@tf_export('debugging.experimental.disable_dump_debug_info')
def disable_dump_debug_info():
    """Disable the currently-enabled debugging dumping.

  If the `enable_dump_debug_info()` method under the same Python namespace
  has been invoked before, calling this method disables it. If no call to
  `enable_dump_debug_info()` has been made, calling this method is a no-op.
  Calling this method more than once is idempotent.
  """
    if hasattr(_state, 'dumping_callback'):
        dump_root = _state.dumping_callback.dump_root
        tfdbg_run_id = _state.dumping_callback.tfdbg_run_id
        debug_events_writer.DebugEventsWriter(dump_root, tfdbg_run_id).Close()
        op_callbacks.remove_op_callback(_state.dumping_callback.callback)
        if _state.dumping_callback.function_callback in function_lib.CONCRETE_FUNCTION_CALLBACKS:
            function_lib.CONCRETE_FUNCTION_CALLBACKS.remove(_state.dumping_callback.function_callback)
        delattr(_state, 'dumping_callback')
        logging.info('Disabled dumping callback in thread %s (dump root: %s)', threading.current_thread().name, dump_root)