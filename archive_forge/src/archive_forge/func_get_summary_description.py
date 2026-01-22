import contextlib
import warnings
from google.protobuf import json_format as _json_format
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.framework.summary_pb2 import SummaryDescription
from tensorflow.core.framework.summary_pb2 import SummaryMetadata as _SummaryMetadata  # pylint: enable=unused-import
from tensorflow.core.util.event_pb2 import Event
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.core.util.event_pb2 import TaggedRunMetadata
from tensorflow.python.distribute import summary_op_util as _distribute_summary_op_util
from tensorflow.python.eager import context as _context
from tensorflow.python.framework import constant_op as _constant_op
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import gen_logging_ops as _gen_logging_ops
from tensorflow.python.ops import gen_summary_ops as _gen_summary_ops  # pylint: disable=unused-import
from tensorflow.python.ops import summary_op_util as _summary_op_util
from tensorflow.python.ops import summary_ops_v2 as _summary_ops_v2
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.summary.writer.writer_cache import FileWriterCache
from tensorflow.python.training import training_util as _training_util
from tensorflow.python.util import compat as _compat
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['summary.get_summary_description'])
def get_summary_description(node_def):
    """Given a TensorSummary node_def, retrieve its SummaryDescription.

  When a Summary op is instantiated, a SummaryDescription of associated
  metadata is stored in its NodeDef. This method retrieves the description.

  Args:
    node_def: the node_def_pb2.NodeDef of a TensorSummary op

  Returns:
    a summary_pb2.SummaryDescription

  Raises:
    ValueError: if the node is not a summary op.

  @compatibility(eager)
  Not compatible with eager execution. To write TensorBoard
  summaries under eager execution, use `tf.contrib.summary` instead.
  @end_compatibility
  """
    if node_def.op != 'TensorSummary':
        raise ValueError("Can't get_summary_description on %s" % node_def.op)
    description_str = _compat.as_str_any(node_def.attr['description'].s)
    summary_description = SummaryDescription()
    _json_format.Parse(description_str, summary_description)
    return summary_description