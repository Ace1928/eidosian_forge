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
def update_dtype(self, attr_name, index, dtype):
    """Changes the type of a given input.

    Args:
      attr_name: The NodeDef attribute containing the type to change.
      index: The index of the input type to change.
      dtype: The type to change to.
    """
    attr = self._node.attr[attr_name]
    num_types = 0
    if attr.HasField('list'):
        types = attr.list.type
        num_types = len(types)
        if num_types > index:
            types[index] = dtype
            return
    elif attr.HasField('type'):
        num_types = 1
        if index == 0:
            attr.type = dtype
            return
    raise ValueError(f'`index` {index:d} is out of range for node({self._node.name}).attr({attr_name}), which has {num_types:d} elements.')