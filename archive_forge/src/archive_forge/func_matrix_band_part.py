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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('linalg.band_part', v1=['linalg.band_part', 'matrix_band_part'])
@deprecated_endpoints('matrix_band_part')
def matrix_band_part(input: _atypes.TensorFuzzingAnnotation[TV_MatrixBandPart_T], num_lower: _atypes.TensorFuzzingAnnotation[TV_MatrixBandPart_Tindex], num_upper: _atypes.TensorFuzzingAnnotation[TV_MatrixBandPart_Tindex], name=None) -> _atypes.TensorFuzzingAnnotation[TV_MatrixBandPart_T]:
    """Copy a tensor setting everything outside a central band in each innermost matrix to zero.

  The `band` part is computed as follows:
  Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
  tensor with the same shape where

  `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.

  The indicator function

  `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
                   (num_upper < 0 || (n-m) <= num_upper)`.

  For example:

  ```
  # if 'input' is [[ 0,  1,  2, 3]
  #                [-1,  0,  1, 2]
  #                [-2, -1,  0, 1]
  #                [-3, -2, -1, 0]],

  tf.linalg.band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
                                         [-1,  0,  1, 2]
                                         [ 0, -1,  0, 1]
                                         [ 0,  0, -1, 0]],

  tf.linalg.band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
                                        [-1,  0,  1, 0]
                                        [-2, -1,  0, 1]
                                        [ 0, -2, -1, 0]]
  ```

  Useful special cases:

  ```
   tf.linalg.band_part(input, 0, -1) ==> Upper triangular part.
   tf.linalg.band_part(input, -1, 0) ==> Lower triangular part.
   tf.linalg.band_part(input, 0, 0) ==> Diagonal.
  ```

  Args:
    input: A `Tensor`. Rank `k` tensor.
    num_lower: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D tensor. Number of subdiagonals to keep. If negative, keep entire
      lower triangle.
    num_upper: A `Tensor`. Must have the same type as `num_lower`.
      0-D tensor. Number of superdiagonals to keep. If negative, keep
      entire upper triangle.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MatrixBandPart', name, input, num_lower, num_upper)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_matrix_band_part((input, num_lower, num_upper, name), None)
            if _result is not NotImplemented:
                return _result
            return matrix_band_part_eager_fallback(input, num_lower, num_upper, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(matrix_band_part, (), dict(input=input, num_lower=num_lower, num_upper=num_upper, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_matrix_band_part((input, num_lower, num_upper, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('MatrixBandPart', input=input, num_lower=num_lower, num_upper=num_upper, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(matrix_band_part, (), dict(input=input, num_lower=num_lower, num_upper=num_upper, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindex', _op._get_attr_type('Tindex'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MatrixBandPart', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result