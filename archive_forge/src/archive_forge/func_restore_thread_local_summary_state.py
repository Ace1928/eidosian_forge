import contextlib
import threading
import weakref
from tensorflow.python import pywrap_tfe
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import shared_variable_creator
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.util import traceback_utils
def restore_thread_local_summary_state(self):
    """Restore thread local summary state from self."""
    summary_state = summary_ops_v2._summary_state
    summary_state.step = self._summary_step
    summary_state.writer = self._summary_writer
    summary_state.is_recording = self._summary_recording
    summary_state.is_recording_distribution_strategy = self._summary_recording_distribution_strategy