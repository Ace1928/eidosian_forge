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
@tf_export('linalg.lu')
def lu(input: _atypes.TensorFuzzingAnnotation[TV_Lu_T], output_idx_type: TV_Lu_output_idx_type=_dtypes.int32, name=None):
    """Computes the LU decomposition of one or more square matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices.

  The input has to be invertible.

  The output consists of two tensors LU and P containing the LU decomposition
  of all input submatrices `[..., :, :]`. LU encodes the lower triangular and
  upper triangular factors.

  For each input submatrix of shape `[M, M]`, L is a lower triangular matrix of
  shape `[M, M]` with unit diagonal whose entries correspond to the strictly lower
  triangular part of LU. U is a upper triangular matrix of shape `[M, M]` whose
  entries correspond to the upper triangular part, including the diagonal, of LU.

  P represents a permutation matrix encoded as a list of indices each between `0`
  and `M-1`, inclusive. If P_mat denotes the permutation matrix corresponding to
  P, then the L, U and P satisfies P_mat * input = L * U.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`, `half`, `complex64`, `complex128`.
      A tensor of shape `[..., M, M]` whose inner-most 2 dimensions form matrices of
      size `[M, M]`.
    output_idx_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (lu, p).

    lu: A `Tensor`. Has the same type as `input`.
    p: A `Tensor` of type `output_idx_type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Lu', name, input, 'output_idx_type', output_idx_type)
            _result = _LuOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_lu((input, output_idx_type, name), None)
            if _result is not NotImplemented:
                return _result
            return lu_eager_fallback(input, output_idx_type=output_idx_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(lu, (), dict(input=input, output_idx_type=output_idx_type, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_lu((input, output_idx_type, name), None)
        if _result is not NotImplemented:
            return _result
    if output_idx_type is None:
        output_idx_type = _dtypes.int32
    output_idx_type = _execute.make_type(output_idx_type, 'output_idx_type')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('Lu', input=input, output_idx_type=output_idx_type, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(lu, (), dict(input=input, output_idx_type=output_idx_type, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'output_idx_type', _op._get_attr_type('output_idx_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Lu', _inputs_flat, _attrs, _result)
    _result = _LuOutput._make(_result)
    return _result