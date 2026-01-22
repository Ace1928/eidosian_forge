import collections
import inspect
import threading
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack
Returns a map (filename, lineno) -> (filename, lineno, function_name).