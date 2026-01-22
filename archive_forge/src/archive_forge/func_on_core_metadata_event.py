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
def on_core_metadata_event(self, event):
    self._event_listener_servicer.toggle_watch()
    core_metadata = json.loads(event.log_message.message)
    if not self._grpc_path:
        grpc_path = core_metadata['grpc_path']
        if grpc_path:
            if grpc_path.startswith('/'):
                grpc_path = grpc_path[1:]
        if self._dump_dir:
            self._dump_dir = os.path.join(self._dump_dir, grpc_path)
            for graph_def, device_name, wall_time in zip(self._cached_graph_defs, self._cached_graph_def_device_names, self._cached_graph_def_wall_times):
                self._write_graph_def(graph_def, device_name, wall_time)
    if self._dump_dir:
        self._write_core_metadata_event(event)
    else:
        self._event_listener_servicer.core_metadata_json_strings.append(event.log_message.message)