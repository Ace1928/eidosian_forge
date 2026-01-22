import collections
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def disable_lower_using_switch_merge(graph_def):
    """Set '_lower_using_switch_merge' attributes to False.

  Sets the attribute to False in the NodeDefs in the main graph and the NodeDefs
  in each function's graph.

  Args:
    graph_def: GraphDef proto.

  Returns:
    GraphDef
  """
    output_graph_def = graph_pb2.GraphDef()
    output_graph_def.CopyFrom(graph_def)

    def disable_control_flow_lowering(node):
        if node.op in _CONTROL_FLOW_OPS:
            node.attr['_lower_using_switch_merge'].b = False
    for node in output_graph_def.node:
        disable_control_flow_lowering(node)
    if output_graph_def.library:
        for func in output_graph_def.library.function:
            for node in func.node_def:
                disable_control_flow_lowering(node)
    return output_graph_def