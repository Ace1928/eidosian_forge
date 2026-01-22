import copy
import re
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import _proto_comparators
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(date=None, instructions=_DEPRECATION_MSG)
@tf_export(v1=['graph_util.tensor_shape_from_node_def_name'])
def tensor_shape_from_node_def_name(graph, input_name):
    """Convenience function to get a shape from a NodeDef's input string."""
    if ':' not in input_name:
        canonical_name = input_name + ':0'
    else:
        canonical_name = input_name
    tensor = graph.get_tensor_by_name(canonical_name)
    shape = tensor.get_shape()
    return shape