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
@tf_export('linalg.solve', v1=['linalg.solve', 'matrix_solve'])
@deprecated_endpoints('matrix_solve')
def matrix_solve(matrix: _atypes.TensorFuzzingAnnotation[TV_MatrixSolve_T], rhs: _atypes.TensorFuzzingAnnotation[TV_MatrixSolve_T], adjoint: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_MatrixSolve_T]:
    """Solves systems of linear equations.

  `Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
  a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
  satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
  If `adjoint` is `True` then each output matrix satisfies
  `adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`, `half`, `complex64`, `complex128`.
      Shape is `[..., M, M]`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
      Shape is `[..., M, K]`.
    adjoint: An optional `bool`. Defaults to `False`.
      Boolean indicating whether to solve with `matrix` or its (block-wise)
      adjoint.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MatrixSolve', name, matrix, rhs, 'adjoint', adjoint)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_matrix_solve((matrix, rhs, adjoint, name), None)
            if _result is not NotImplemented:
                return _result
            return matrix_solve_eager_fallback(matrix, rhs, adjoint=adjoint, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(matrix_solve, (), dict(matrix=matrix, rhs=rhs, adjoint=adjoint, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_matrix_solve((matrix, rhs, adjoint, name), None)
        if _result is not NotImplemented:
            return _result
    if adjoint is None:
        adjoint = False
    adjoint = _execute.make_bool(adjoint, 'adjoint')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('MatrixSolve', matrix=matrix, rhs=rhs, adjoint=adjoint, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(matrix_solve, (), dict(matrix=matrix, rhs=rhs, adjoint=adjoint, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('adjoint', _op._get_attr_bool('adjoint'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MatrixSolve', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result