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
def query_origin_stack(self):
    """Query the stack of the origin of the execution call.

    Returns:
      A `list` of all tracebacks. Each item corresponds to an execution call,
        i.e., a `SendTracebacks` request. Each item is a `list` of 3-tuples:
        (filename, lineno, function_name).
    """
    ret = []
    for stack, id_to_string in zip(self._origin_stacks, self._origin_id_to_strings):
        ret.append(self._code_def_to_traceback(stack, id_to_string))
    return ret