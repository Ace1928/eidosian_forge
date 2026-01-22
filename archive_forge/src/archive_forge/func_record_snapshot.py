import collections
import copy
import gc
from tensorflow.python.framework import _python_memory_checker_helper
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
def record_snapshot(self):
    _python_memory_checker_helper.mark_stack_trace_and_call(self._record_snapshot)