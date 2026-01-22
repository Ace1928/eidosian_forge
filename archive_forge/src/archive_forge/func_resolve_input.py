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
def resolve_input(self, input_name):
    """Resolves an input into its _EndPoint.

    A NodeDef's input name can refer to either global NodeDefs (in the
    GraphDef's node list), a NodeDef in a function's node list, or a Function
    (in the GraphDef's function library). The name can also carry semantic
    information, depending on whether it starts with "^". This method handles
    all that logic in order to find the object to which the input name refers
    to.

    Args:
      input_name: The input name to resolve.

    Returns:
      The object referred to by 'input_name'.
    """
    name_elts = input_name.split(':')
    source_name = name_elts[0]
    if source_name[0] == '^':
        source_name = source_name[1:]
    source_index = 0
    if len(name_elts) > 1 and name_elts[-1].isnumeric():
        source_index = int(name_elts[-1])
    if self._function is None:
        return _EndPoint(self._enclosing_graph.nodes[source_name], source_index)
    if source_index != 0 or source_name in self._function.nodes:
        return _EndPoint(self._function.nodes[source_name], source_index)
    inputs = [i.name for i in self._function.function.signature.input_arg]
    return _EndPoint(self._function, inputs.index(source_name))