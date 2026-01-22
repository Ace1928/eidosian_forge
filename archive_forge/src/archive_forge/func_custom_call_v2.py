from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.numpy_ops import np_utils
def custom_call_v2(call_target_name, operands, result_specs, backend_config=None, has_side_effect=None, name=None):
    """Emits an HLO `CustomCall` operation with multiple outputs.

  See `CustomCall` specification at
    https://tensorflow.org/xla/operation_semantics#customcall,
  and `mhlo.custom_call` specification at
    https://tensorflow.org/mlir/hlo_ops#mhlocustom_call_mlirmhlocustomcallop.

  Args:
    call_target_name: Name of the user function. The function signature must
      conform to version 3 of the API, see
      `API_VERSION_STATUS_RETURNING_UNIFIED`. All operands and results assumed
      to be in the default layout.
    operands: A sequence of tensors with possibly different types.
    result_specs: A sequence of tensor specs for all results.
    backend_config: A string that encodes a metadata for the backend. Empty
      string by default.
    has_side_effect: Indicates whether the custom call has side effects. `False`
      by default.
    name: Optional name of the operation.

  Returns:
    A tuple of output tensors.
  """
    return gen_xla_ops.xla_custom_call_v2(operands=operands, call_target_name=call_target_name, backend_config='' if backend_config is None else backend_config, has_side_effect=False if has_side_effect is None else has_side_effect, result_dtypes=tuple((spec.dtype for spec in result_specs)), result_shapes=tuple((spec.shape for spec in result_specs)), name=name)