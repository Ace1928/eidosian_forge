import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def symbolic_tensor_id(self, graph_id, op_name, output_slot):
    """Get the ID of a symbolic tensor.

    Args:
      graph_id: The ID of the immediately-enclosing graph.
      op_name: Name of the op.
      output_slot: Output slot as an int.

    Returns:
      The ID of the symbolic tensor as an int.
    """
    return self._graph_by_id[graph_id].get_tensor_id(op_name, output_slot)