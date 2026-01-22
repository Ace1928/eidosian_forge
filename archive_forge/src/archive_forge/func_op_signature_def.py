from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import utils_impl as utils
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def op_signature_def(op, key):
    """Creates a signature def with the output pointing to an op.

  Note that op isn't strictly enforced to be an Op object, and may be a Tensor.
  It is recommended to use the build_signature_def() function for Tensors.

  Args:
    op: An Op (or possibly Tensor).
    key: Key to graph element in the SignatureDef outputs.

  Returns:
    A SignatureDef with a single output pointing to the op.
  """
    return build_signature_def(outputs={key: utils.build_tensor_info_from_op(op)})