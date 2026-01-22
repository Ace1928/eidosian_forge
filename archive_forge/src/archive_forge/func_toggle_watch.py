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
def toggle_watch(self):
    for watch_key in self._toggle_watch_state:
        node_name, output_slot, debug_op = watch_key
        if self._toggle_watch_state[watch_key]:
            self.request_unwatch(node_name, output_slot, debug_op)
        else:
            self.request_watch(node_name, output_slot, debug_op)
        self._toggle_watch_state[watch_key] = not self._toggle_watch_state[watch_key]