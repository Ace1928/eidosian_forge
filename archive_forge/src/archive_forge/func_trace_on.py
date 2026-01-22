import abc
import collections
import functools
import os
import re
import threading
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import profiler as _profiler
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import resource
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export('summary.trace_on', v1=[])
def trace_on(graph=True, profiler=False):
    """Starts a trace to record computation graphs and profiling information.

  Must be invoked in eager mode.

  When enabled, TensorFlow runtime will collect information that can later be
  exported and consumed by TensorBoard. The trace is activated across the entire
  TensorFlow runtime and affects all threads of execution.

  To stop the trace and export the collected information, use
  `tf.summary.trace_export`. To stop the trace without exporting, use
  `tf.summary.trace_off`.

  Args:
    graph: If True, enables collection of executed graphs. It includes ones from
        tf.function invocation and ones from the legacy graph mode. The default
        is True.
    profiler: If True, enables the advanced profiler. Enabling profiler
        implicitly enables the graph collection. The profiler may incur a high
        memory overhead. The default is False.

  """
    if ops.inside_function():
        logging.warn('Cannot enable trace inside a tf.function.')
        return
    if not context.executing_eagerly():
        logging.warn('Must enable trace in eager mode.')
        return
    global _current_trace_context
    with _current_trace_context_lock:
        if _current_trace_context:
            logging.warn('Trace already enabled')
            return
        if graph and (not profiler):
            context.context().enable_graph_collection()
        if profiler:
            context.context().enable_run_metadata()
            _profiler.start()
        _current_trace_context = _TraceContext(graph=graph, profiler=profiler)