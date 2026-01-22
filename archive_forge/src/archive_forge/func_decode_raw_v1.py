from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_config
from tensorflow.python.ops.gen_parsing_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['decode_raw', 'io.decode_raw'])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, 'bytes is deprecated, use input_bytes instead', 'bytes')
def decode_raw_v1(input_bytes=None, out_type=None, little_endian=True, name=None, bytes=None):
    """Convert raw byte strings into tensors.

  Args:
    input_bytes:
      Each element of the input Tensor is converted to an array of bytes.
    out_type:
      `DType` of the output. Acceptable types are `half`, `float`, `double`,
      `int32`, `uint16`, `uint8`, `int16`, `int8`, `int64`.
    little_endian:
      Whether the `input_bytes` data is in little-endian format. Data will be
      converted into host byte order if necessary.
    name: A name for the operation (optional).
    bytes: Deprecated parameter. Use `input_bytes` instead.

  Returns:
    A `Tensor` object storing the decoded bytes.
  """
    input_bytes = deprecation.deprecated_argument_lookup('input_bytes', input_bytes, 'bytes', bytes)
    if out_type is None:
        raise ValueError("decode_raw_v1() missing 1 positional argument: 'out_type'")
    return gen_parsing_ops.decode_raw(input_bytes, out_type, little_endian=little_endian, name=name)