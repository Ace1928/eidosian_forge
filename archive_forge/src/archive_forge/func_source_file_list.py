import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def source_file_list(self):
    """Get a list of source files known to the debugger data reader.

    Returns:
      A tuple of `(host_name, file_path)` tuples.
    """
    return tuple(self._host_name_file_path_to_offset.keys())