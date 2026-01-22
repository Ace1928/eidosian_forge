import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def outermost_graphs(self):
    """Get the number of outer most graphs read so far."""
    return [graph for graph in self._graph_by_id.values() if not graph.outer_graph_id]