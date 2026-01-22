import collections
import os
import re
import site
import traceback
from tensorflow.python.util import tf_stack
def create_graph_debug_info_def(func_named_operations):
    """Construct and returns a `GraphDebugInfo` protocol buffer.

  Args:
    func_named_operations: An iterable of (func_name, op.Operation) tuples
      where the Operation instances have a _traceback members. The func_name
      should be the empty string for operations in the top-level Graph.

  Returns:
    GraphDebugInfo protocol buffer.

  Raises:
    TypeError: If the arguments are not of the correct proto buffer type.
  """
    builder = tf_stack.GraphDebugInfoBuilder()
    for func_name, op in func_named_operations:
        if op.traceback is None:
            continue
        builder.AccumulateStackTrace(func_name, op.name, _compute_useful_frames(op.traceback, 10))
    return builder.Build()