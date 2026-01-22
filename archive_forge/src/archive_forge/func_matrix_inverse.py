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
@tf_export('linalg.inv', v1=['linalg.inv', 'matrix_inverse'])
@deprecated_endpoints('matrix_inverse')
def matrix_inverse(input: _atypes.TensorFuzzingAnnotation[TV_MatrixInverse_T], adjoint: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_MatrixInverse_T]:
    """Computes the inverse of one or more square invertible matrices or their adjoints (conjugate transposes).

  
  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a tensor of the same shape as the input
  containing the inverse for all input submatrices `[..., :, :]`.

  The op uses LU decomposition with partial pivoting to compute the inverses.

  If a matrix is not invertible there is no guarantee what the op does. It
  may detect the condition and raise an exception or it may simply return a
  garbage result.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`, `half`, `complex64`, `complex128`.
      Shape is `[..., M, M]`.
    adjoint: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MatrixInverse', name, input, 'adjoint', adjoint)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_matrix_inverse((input, adjoint, name), None)
            if _result is not NotImplemented:
                return _result
            return matrix_inverse_eager_fallback(input, adjoint=adjoint, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(matrix_inverse, (), dict(input=input, adjoint=adjoint, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_matrix_inverse((input, adjoint, name), None)
        if _result is not NotImplemented:
            return _result
    if adjoint is None:
        adjoint = False
    adjoint = _execute.make_bool(adjoint, 'adjoint')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('MatrixInverse', input=input, adjoint=adjoint, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(matrix_inverse, (), dict(input=input, adjoint=adjoint, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('adjoint', _op._get_attr_bool('adjoint'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MatrixInverse', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result