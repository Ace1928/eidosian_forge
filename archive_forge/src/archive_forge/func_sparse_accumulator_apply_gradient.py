import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def sparse_accumulator_apply_gradient(handle: _atypes.TensorFuzzingAnnotation[_atypes.String], local_step: _atypes.TensorFuzzingAnnotation[_atypes.Int64], gradient_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], gradient_values: _atypes.TensorFuzzingAnnotation[TV_SparseAccumulatorApplyGradient_dtype], gradient_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], has_known_shape: bool, name=None):
    """Applies a sparse gradient to a given accumulator.

  Does not add if local_step is smaller than the accumulator's
  global_step.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a accumulator.
    local_step: A `Tensor` of type `int64`.
      The local_step value at which the sparse gradient was computed.
    gradient_indices: A `Tensor` of type `int64`.
      Indices of the sparse gradient to be accumulated. Must be a
      vector.
    gradient_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Values are the non-zero slices of the gradient, and must have
      the same first dimension as indices, i.e., the nnz represented by indices and
      values must be consistent.
    gradient_shape: A `Tensor` of type `int64`.
      Shape of the sparse gradient to be accumulated.
    has_known_shape: A `bool`.
      Boolean indicating whether gradient_shape is unknown, in which
      case the input is ignored during validation.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("sparse_accumulator_apply_gradient op does not support eager execution. Arg 'handle' is a ref.")
    has_known_shape = _execute.make_bool(has_known_shape, 'has_known_shape')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseAccumulatorApplyGradient', handle=handle, local_step=local_step, gradient_indices=gradient_indices, gradient_values=gradient_values, gradient_shape=gradient_shape, has_known_shape=has_known_shape, name=name)
    return _op