from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
Returns the element in the graph described by a TensorInfo proto.

  Args:
    tensor_info: A TensorInfo proto describing an Op or Tensor by name.
    graph: The tf.Graph in which tensors are looked up. If None, the current
      default graph is used.
    import_scope: If not None, names in `tensor_info` are prefixed with this
      string before lookup.

  Returns:
    Op or tensor in `graph` described by `tensor_info`.

  Raises:
    KeyError: If `tensor_info` does not correspond to an op or tensor in `graph`
  