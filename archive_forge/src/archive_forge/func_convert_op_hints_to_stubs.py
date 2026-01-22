import collections as _collections
import copy as _copy
import json as _json
import uuid as _uuid
from tensorflow.core.framework import attr_value_pb2 as _attr_value_pb2
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.core.framework import node_def_pb2 as _node_def_pb2
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.framework.graph_util_impl import _bfs_for_reachable_nodes
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.util import compat as _compat
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export as _tf_export
@_tf_export(v1=['lite.experimental.convert_op_hints_to_stubs'])
@_deprecation.deprecated(None, 'Please follow instructions under https://www.tensorflow.org/lite/convert/operation_fusion for operationfusion in tflite.')
def convert_op_hints_to_stubs(session=None, graph_def=None, write_callback=lambda graph_def, comments: None):
    """Converts a graphdef with LiteOp hints into stub operations.

  This is used to prepare for toco conversion of complex intrinsic usages.
  Note: only one of session or graph_def should be used, not both.

  Args:
    session: A TensorFlow session that contains the graph to convert.
    graph_def: A graph def that we should convert.
    write_callback: A function pointer that can be used to write intermediate
      steps of graph transformation (optional).

  Returns:
    A new graphdef with all ops contained in OpHints being replaced by
    a single op call with the right parameters.
  Raises:
    ValueError: If both session and graph_def are provided.
  """
    if session is not None and graph_def is not None:
        raise ValueError('Provide only one of session and graph_def.')
    if session is not None:
        return _convert_op_hints_to_stubs_helper(session.graph_def, write_callback)
    elif graph_def is not None:
        return _convert_op_hints_to_stubs_helper(graph_def, write_callback)
    else:
        raise ValueError('Must specify session or graph_def as input.')