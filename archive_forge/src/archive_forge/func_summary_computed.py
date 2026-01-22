import contextlib
import os
import time
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as _summary
from tensorflow.python.training import coordinator
from tensorflow.python.training import saver as saver_mod
from tensorflow.python.training import session_manager as session_manager_mod
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def summary_computed(self, sess, summary, global_step=None):
    """Indicate that a summary was computed.

    Args:
      sess: A `Session` object.
      summary: A Summary proto, or a string holding a serialized summary proto.
      global_step: Int. global step this summary is associated with. If `None`,
        it will try to fetch the current step.

    Raises:
      TypeError: if 'summary' is not a Summary proto or a string.
      RuntimeError: if the Supervisor was created without a `logdir`.
    """
    if not self._summary_writer:
        raise RuntimeError('Writing a summary requires a summary writer.')
    if global_step is None and self.global_step is not None:
        global_step = training_util.global_step(sess, self.global_step)
    self._summary_writer.add_summary(summary, global_step)