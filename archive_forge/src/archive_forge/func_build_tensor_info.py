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
@tf_export(v1=['saved_model.build_tensor_info', 'saved_model.utils.build_tensor_info'])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def build_tensor_info(tensor):
    """Utility function to build TensorInfo proto from a Tensor.

  Args:
    tensor: Tensor or SparseTensor whose name, dtype and shape are used to
        build the TensorInfo. For SparseTensors, the names of the three
        constituent Tensors are used.

  Returns:
    A TensorInfo protocol buffer constructed based on the supplied argument.

  Raises:
    RuntimeError: If eager execution is enabled.

  @compatibility(TF2)
  This API is not compatible with eager execution as `tensor` needs to be a
  graph tensor, and there is no replacement for it in TensorFlow 2.x. To start
  writing programs using TensorFlow 2.x, please refer to the [Effective
  TensorFlow 2](https://www.tensorflow.org/guide/effective_tf2) guide.
  @end_compatibility
  """
    if context.executing_eagerly():
        raise RuntimeError('`build_tensor_info` is not supported in eager execution.')
    return build_tensor_info_internal(tensor)