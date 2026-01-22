import datetime
import os
import threading
from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
@deprecated('2020-07-01', '`tf.python.eager.profiler` has deprecated, use `tf.profiler` instead.')
def maybe_create_event_file(logdir):
    """Create an empty event file if not already exists.

  This event file indicates that we have a plugins/profile/ directory in the
  current logdir.

  Args:
    logdir: log directory.
  """
    for file_name in gfile.ListDirectory(logdir):
        if file_name.endswith(_EVENT_FILE_SUFFIX):
            return
    event_writer = _pywrap_events_writer.EventsWriter(compat.as_bytes(os.path.join(logdir, 'events')))
    event_writer.InitWithSuffix(compat.as_bytes(_EVENT_FILE_SUFFIX))