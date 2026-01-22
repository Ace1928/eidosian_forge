from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect
@dispatch.dispatch_for_unary_elementwise_apis(ragged_tensor.Ragged)
def ragged_unary_elementwise_op(op, x):
    """Unary elementwise api handler for RaggedTensors."""
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x)
    return x.with_values(op(x.values))