import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def convert_to_eager_tensor(value, ctx, dtype=None):
    """Converts the given `value` to an `EagerTensor`.

  Note that this function could return cached copies of created constants for
  performance reasons.

  Args:
    value: value to convert to EagerTensor.
    ctx: value of context.context().
    dtype: optional desired dtype of the converted EagerTensor.

  Returns:
    EagerTensor created from value.

  Raises:
    TypeError: if `dtype` is not compatible with the type of t.
  """
    if isinstance(value, np.ndarray):
        value = value.copy()
    if isinstance(value, ops.EagerTensor):
        if dtype is not None and value.dtype != dtype:
            raise TypeError(f'Expected tensor {value} with dtype {dtype!r}, but got dtype {value.dtype!r}.')
        return value
    if dtype is not None:
        try:
            dtype = dtype.as_datatype_enum
        except AttributeError:
            dtype = dtypes.as_dtype(dtype).as_datatype_enum
    ctx.ensure_initialized()
    return ops.EagerTensor(value, ctx.device_name, dtype)