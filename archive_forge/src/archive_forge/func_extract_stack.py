import collections
import inspect
import threading
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack
def extract_stack(stacklevel=1):
    """An eager-friendly alternative to traceback.extract_stack.

  Args:
    stacklevel: number of initial frames to skip when producing the stack.

  Returns:
    A list-like FrameSummary containing StackFrame-like objects, which are
    namedtuple-like objects with the following fields: filename, lineno, name,
    line, meant to masquerade as traceback.FrameSummary objects.
  """
    thread_key = _get_thread_key()
    return _tf_stack.extract_stack(_source_mapper_stacks[thread_key][-1].internal_map, _source_filter_stacks[thread_key][-1].internal_set, stacklevel)