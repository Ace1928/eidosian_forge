import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def get_op_consumers(self, src_op_name):
    """Get all the downstream consumers of this op.

    Only data (non-control) edges are tracked.

    Args:
      src_op_name: Name of the op providing the tensor being consumed.

    Returns:
      A list of (src_slot, dst_op_name, dst_slot) tuples. In each item of
      the list:
        src_slot: 0-based output slot of the op of which the output tensor
          is being consumed.
        dst_op_name: Name of the consuming op (e.g., "Conv2D_3/BiasAdd")
        dst_slot: 0-based input slot of the consuming op that receives
          the tensor from this op.
    """
    return self._op_consumers[src_op_name]