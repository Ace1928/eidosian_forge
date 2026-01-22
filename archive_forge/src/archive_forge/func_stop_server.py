import collections
import json
import queue
import threading
import time
from concurrent import futures
import grpc
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_service_pb2_grpc
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def stop_server(self, grace=1.0):
    """Request server stopping.

    Once stopped, server cannot be stopped or started again. This method is
    non-blocking. Call `wait()` on the returned event to block until the server
    has completely stopped.

    Args:
      grace: Grace period in seconds to be used when calling `server.stop()`.

    Raises:
      ValueError: If server stop has already been requested, or if the server
        has not started running yet.

    Returns:
      A threading.Event that will be set when the server has completely stopped.
    """
    self._server_lock.acquire()
    try:
        if not self._server_started:
            raise ValueError('Server has not started running')
        if self._stop_requested:
            raise ValueError('Server has already stopped')
        self._stop_requested = True
        return self.server.stop(grace=grace)
    finally:
        self._server_lock.release()