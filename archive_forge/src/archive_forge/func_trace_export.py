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
@tf_export('summary.trace_export', v1=[])
def trace_export(name, step=None, profiler_outdir=None):
    """Stops and exports the active trace as a Summary and/or profile file.

  Stops the trace and exports all metadata collected during the trace to the
  default SummaryWriter, if one has been set.

  Args:
    name: A name for the summary to be written.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    profiler_outdir: Output directory for profiler. It is required when profiler
      is enabled when trace was started. Otherwise, it is ignored.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
    global _current_trace_context
    if ops.inside_function():
        logging.warn('Cannot export trace inside a tf.function.')
        return
    if not context.executing_eagerly():
        logging.warn('Can only export trace while executing eagerly.')
        return
    with _current_trace_context_lock:
        if _current_trace_context is None:
            raise ValueError('Must enable trace before export through tf.summary.trace_on.')
        graph, profiler = _current_trace_context
        if profiler and profiler_outdir is None:
            raise ValueError('Argument `profiler_outdir` is not specified.')
    run_meta = context.context().export_run_metadata()
    if graph and (not profiler):
        run_metadata_graphs(name, run_meta, step)
    else:
        run_metadata(name, run_meta, step)
    if profiler:
        _profiler.save(profiler_outdir, _profiler.stop())
    trace_off()