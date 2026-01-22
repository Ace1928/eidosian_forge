import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def graph_op_digests(self, op_type=None):
    """Get the list of the digests for graph-op creation so far.

    Args:
      op_type: Optional op type to filter the creation events with.

    Returns:
      A list of `GraphOpCreationDigest` objects.
    """
    if op_type is not None:
        return [digest for digest in self._graph_op_digests if digest.op_type == op_type]
    else:
        return self._graph_op_digests