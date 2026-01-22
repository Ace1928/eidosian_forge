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
@property
def breakpoints(self):
    """Get a set of the currently-activated breakpoints.

    Returns:
      A `set` of 3-tuples: (node_name, output_slot, debug_op), e.g.,
        {("MatMul", 0, "DebugIdentity")}.
    """
    return self._breakpoints