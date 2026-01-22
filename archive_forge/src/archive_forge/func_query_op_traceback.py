import collections
import errno
import functools
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import portpicker
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import grpc_debug_server
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.util import compat
def query_op_traceback(self, op_name):
    """Query the traceback of an op.

    Args:
      op_name: Name of the op to query.

    Returns:
      The traceback of the op, as a list of 3-tuples:
        (filename, lineno, function_name)

    Raises:
      ValueError: If the op cannot be found in the tracebacks received by the
        server so far.
    """
    for op_log_proto in self._graph_tracebacks:
        for log_entry in op_log_proto.log_entries:
            if log_entry.name == op_name:
                return self._code_def_to_traceback(log_entry.code_def, op_log_proto.id_to_string)
    raise ValueError("Op '%s' does not exist in the tracebacks received by the debug server." % op_name)