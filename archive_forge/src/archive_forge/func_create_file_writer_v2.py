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
@tf_export('summary.create_file_writer', v1=[])
def create_file_writer_v2(logdir, max_queue=None, flush_millis=None, filename_suffix=None, name=None, experimental_trackable=False, experimental_mesh=None):
    """Creates a summary file writer for the given log directory.

  Args:
    logdir: a string specifying the directory in which to write an event file.
    max_queue: the largest number of summaries to keep in a queue; will flush
      once the queue gets bigger than this. Defaults to 10.
    flush_millis: the largest interval between flushes. Defaults to 120,000.
    filename_suffix: optional suffix for the event file name. Defaults to `.v2`.
    name: a name for the op that creates the writer.
    experimental_trackable: a boolean that controls whether the returned writer
      will be a `TrackableResource`, which makes it compatible with SavedModel
      when used as a `tf.Module` property.
    experimental_mesh: a `tf.experimental.dtensor.Mesh` instance. When running
      with DTensor, the mesh (experimental_mesh.host_mesh()) will be used for
      bringing all the DTensor logging from accelerator to CPU mesh.

  Returns:
    A SummaryWriter object.
  """
    if logdir is None:
        raise ValueError('Argument `logdir` cannot be None')
    inside_function = ops.inside_function()
    with ops.name_scope(name, 'create_file_writer') as scope, ops.device('cpu:0'):
        with ops.init_scope():
            if context.executing_eagerly():
                _check_create_file_writer_args(inside_function, logdir=logdir, max_queue=max_queue, flush_millis=flush_millis, filename_suffix=filename_suffix)
            logdir = ops.convert_to_tensor(logdir, dtype=dtypes.string)
            if max_queue is None:
                max_queue = constant_op.constant(10)
            if flush_millis is None:
                flush_millis = constant_op.constant(2 * 60 * 1000)
            if filename_suffix is None:
                filename_suffix = constant_op.constant('.v2')

            def create_fn():
                if context.executing_eagerly():
                    shared_name = context.anonymous_name()
                else:
                    shared_name = ops.name_from_scope_name(scope)
                return gen_summary_ops.summary_writer(shared_name=shared_name, name=name)
            init_op_fn = functools.partial(gen_summary_ops.create_summary_file_writer, logdir=logdir, max_queue=max_queue, flush_millis=flush_millis, filename_suffix=filename_suffix)
            if experimental_trackable:
                return _TrackableResourceSummaryWriter(create_fn=create_fn, init_op_fn=init_op_fn, mesh=experimental_mesh)
            else:
                return _ResourceSummaryWriter(create_fn=create_fn, init_op_fn=init_op_fn, mesh=experimental_mesh)