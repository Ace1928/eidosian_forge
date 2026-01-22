import os
import sys
from tensorflow.core.profiler import tfprof_log_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.profiler.internal import flops_registry  # pylint: disable=unused-import
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['profiler.write_op_log'])
def write_op_log(graph, log_dir, op_log=None, run_meta=None, add_trace=True):
    """Log provided 'op_log', and add additional model information below.

    The API also assigns ops in tf.compat.v1.trainable_variables() an op type
    called '_trainable_variables'.
    The API also logs 'flops' statistics for ops with op.RegisterStatistics()
    defined. flops calculation depends on Tensor shapes defined in 'graph',
    which might not be complete. 'run_meta', if provided, completes the shape
    information with best effort.

  Args:
    graph: tf.Graph. If None and eager execution is not enabled, use
        default graph.
    log_dir: directory to write the log file.
    op_log: (Optional) OpLogProto proto to be written. If not provided, an new
        one is created.
    run_meta: (Optional) RunMetadata proto that helps flops computation using
        run time shape information.
    add_trace: Whether to add python code trace information.
        Used to support "code" view.
  """
    if not graph and (not context.executing_eagerly()):
        graph = ops.get_default_graph()
    op_log = merge_default_with_oplog(graph, op_log, run_meta, add_trace)
    with gfile.Open(os.path.join(log_dir, 'tfprof_log'), 'w') as log:
        log.write(op_log.SerializeToString())