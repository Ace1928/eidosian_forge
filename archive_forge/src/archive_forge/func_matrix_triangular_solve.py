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
def matrix_triangular_solve(matrix: _atypes.TensorFuzzingAnnotation[TV_MatrixTriangularSolve_T], rhs: _atypes.TensorFuzzingAnnotation[TV_MatrixTriangularSolve_T], lower: bool=True, adjoint: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_MatrixTriangularSolve_T]:
    """Solves systems of linear equations with upper or lower triangular matrices by backsubstitution.

  
  `matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
  square matrices. If `lower` is `True` then the strictly upper triangular part
  of each inner-most matrix is assumed to be zero and not accessed.
  If `lower` is False then the strictly lower triangular part of each inner-most
  matrix is assumed to be zero and not accessed.
  `rhs` is a tensor of shape `[..., M, N]`.

  The output is a tensor of shape `[..., M, N]`. If `adjoint` is
  `True` then the innermost matrices in `output` satisfy matrix equations
  `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
  If `adjoint` is `False` then the strictly then the  innermost matrices in
  `output` satisfy matrix equations
  `adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`.

  Note, the batch shapes for the inputs only need to broadcast.

  Example:
  ```python

  a = tf.constant([[3,  0,  0,  0],
                   [2,  1,  0,  0],
                   [1,  0,  1,  0],
                   [1,  1,  1,  1]], dtype=tf.float32)

  b = tf.constant([[4],
                   [2],
                   [4],
                   [2]], dtype=tf.float32)

  x = tf.linalg.triangular_solve(a, b, lower=True)
  x
  # <tf.Tensor: shape=(4, 1), dtype=float32, numpy=
  # array([[ 1.3333334 ],
  #        [-0.66666675],
  #        [ 2.6666665 ],
  #        [-1.3333331 ]], dtype=float32)>

  # in python3 one can use `a@x`
  tf.matmul(a, x)
  # <tf.Tensor: shape=(4, 1), dtype=float32, numpy=
  # array([[4.       ],
  #        [2.       ],
  #        [4.       ],
  #        [1.9999999]], dtype=float32)>
  ```

  Args:
    matrix: A `Tensor`. Must be one of the following types: `bfloat16`, `float64`, `float32`, `half`, `complex64`, `complex128`.
      Shape is `[..., M, M]`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
      Shape is `[..., M, K]`.
    lower: An optional `bool`. Defaults to `True`.
      Boolean indicating whether the innermost matrices in `matrix` are
      lower or upper triangular.
    adjoint: An optional `bool`. Defaults to `False`.
      Boolean indicating whether to solve with `matrix` or its (block-wise)
               adjoint.

      @compatibility(numpy)
      Equivalent to scipy.linalg.solve_triangular
      @end_compatibility
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MatrixTriangularSolve', name, matrix, rhs, 'lower', lower, 'adjoint', adjoint)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return matrix_triangular_solve_eager_fallback(matrix, rhs, lower=lower, adjoint=adjoint, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if lower is None:
        lower = True
    lower = _execute.make_bool(lower, 'lower')
    if adjoint is None:
        adjoint = False
    adjoint = _execute.make_bool(adjoint, 'adjoint')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MatrixTriangularSolve', matrix=matrix, rhs=rhs, lower=lower, adjoint=adjoint, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('lower', _op._get_attr_bool('lower'), 'adjoint', _op._get_attr_bool('adjoint'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MatrixTriangularSolve', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result