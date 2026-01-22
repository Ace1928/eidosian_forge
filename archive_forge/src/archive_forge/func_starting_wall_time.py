import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def starting_wall_time(self):
    """Wall timestamp for when the debugged TensorFlow program started.

    Returns:
      Stating wall time as seconds since the epoch, as a `float`.
    """
    return self._reader.starting_wall_time()