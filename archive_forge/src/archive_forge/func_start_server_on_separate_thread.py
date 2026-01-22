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
def start_server_on_separate_thread(dump_to_filesystem=True, server_start_delay_sec=0.0, poll_server=False, blocking=True, toggle_watch_on_core_metadata=None):
    """Create a test gRPC debug server and run on a separate thread.

  Args:
    dump_to_filesystem: (bool) whether the debug server will dump debug data
      to the filesystem.
    server_start_delay_sec: (float) amount of time (in sec) to delay the server
      start up for.
    poll_server: (bool) whether the server will be polled till success on
      startup.
    blocking: (bool) whether the server should be started in a blocking mode.
    toggle_watch_on_core_metadata: A list of
        (node_name, output_slot, debug_op) tuples to toggle the
        watchpoint status during the on_core_metadata calls (optional).

  Returns:
    server_port: (int) Port on which the server runs.
    debug_server_url: (str) grpc:// URL to the server.
    server_dump_dir: (str) The debug server's dump directory.
    server_thread: The server Thread object.
    server: The `EventListenerTestServicer` object.

  Raises:
    ValueError: If polling the server process for ready state is not successful
      within maximum polling count.
  """
    server_port = portpicker.pick_unused_port()
    debug_server_url = 'grpc://localhost:%d' % server_port
    server_dump_dir = tempfile.mkdtemp() if dump_to_filesystem else None
    server = EventListenerTestServicer(server_port=server_port, dump_dir=server_dump_dir, toggle_watch_on_core_metadata=toggle_watch_on_core_metadata)

    def delay_then_run_server():
        time.sleep(server_start_delay_sec)
        server.run_server(blocking=blocking)
    server_thread = threading.Thread(target=delay_then_run_server)
    server_thread.start()
    if poll_server:
        if not _poll_server_till_success(50, 0.2, debug_server_url, server_dump_dir, server, gpu_memory_fraction=0.1):
            raise ValueError('Failed to start test gRPC debug server at port %d' % server_port)
        server.clear_data()
    return (server_port, debug_server_url, server_dump_dir, server_thread, server)