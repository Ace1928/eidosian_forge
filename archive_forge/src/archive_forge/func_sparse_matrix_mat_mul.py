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
def sparse_matrix_mat_mul(a: _atypes.TensorFuzzingAnnotation[_atypes.Variant], b: _atypes.TensorFuzzingAnnotation[TV_SparseMatrixMatMul_T], transpose_a: bool=False, transpose_b: bool=False, adjoint_a: bool=False, adjoint_b: bool=False, transpose_output: bool=False, conjugate_output: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_SparseMatrixMatMul_T]:
    """Matrix-multiplies a sparse matrix with a dense matrix.

  Returns a dense matrix.
  For inputs A and B, where A is CSR and B is dense; this op returns a dense C;

  If transpose_output is false, returns:
  ```
    C = A . B
  ```

  If transpose_output is `true`, returns:
  ```
    C = transpose(A . B) = transpose(B) . transpose(A)
  ```
  where the transposition is performed along the two innermost (matrix)
  dimensions.

  If conjugate_output is `true`, returns:
  ```
    C = conjugate(A . B) = conjugate(A) . conjugate(B)
  ```

  If both conjugate_output and transpose_output are `true`, returns:
  ```
    C = conjugate(transpose(A . B)) = conjugate(transpose(B)) .
                                      conjugate(transpose(A))
  ```

  Args:
    a: A `Tensor` of type `variant`. A CSRSparseMatrix.
    b: A `Tensor`. A dense tensor.
    transpose_a: An optional `bool`. Defaults to `False`.
      Indicates whether `a` should be transposed.
    transpose_b: An optional `bool`. Defaults to `False`.
      Indicates whether `b` should be transposed.
    adjoint_a: An optional `bool`. Defaults to `False`.
      Indicates whether `a` should be conjugate-transposed.
    adjoint_b: An optional `bool`. Defaults to `False`.
      Indicates whether `b` should be conjugate-transposed.
    transpose_output: An optional `bool`. Defaults to `False`.
      Transposes the product of `a` and `b`.
    conjugate_output: An optional `bool`. Defaults to `False`.
      Conjugates the product of `a` and `b`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `b`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseMatrixMatMul', name, a, b, 'transpose_a', transpose_a, 'transpose_b', transpose_b, 'adjoint_a', adjoint_a, 'adjoint_b', adjoint_b, 'transpose_output', transpose_output, 'conjugate_output', conjugate_output)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_matrix_mat_mul_eager_fallback(a, b, transpose_a=transpose_a, transpose_b=transpose_b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, transpose_output=transpose_output, conjugate_output=conjugate_output, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if transpose_a is None:
        transpose_a = False
    transpose_a = _execute.make_bool(transpose_a, 'transpose_a')
    if transpose_b is None:
        transpose_b = False
    transpose_b = _execute.make_bool(transpose_b, 'transpose_b')
    if adjoint_a is None:
        adjoint_a = False
    adjoint_a = _execute.make_bool(adjoint_a, 'adjoint_a')
    if adjoint_b is None:
        adjoint_b = False
    adjoint_b = _execute.make_bool(adjoint_b, 'adjoint_b')
    if transpose_output is None:
        transpose_output = False
    transpose_output = _execute.make_bool(transpose_output, 'transpose_output')
    if conjugate_output is None:
        conjugate_output = False
    conjugate_output = _execute.make_bool(conjugate_output, 'conjugate_output')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseMatrixMatMul', a=a, b=b, transpose_a=transpose_a, transpose_b=transpose_b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, transpose_output=transpose_output, conjugate_output=conjugate_output, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'transpose_a', _op._get_attr_bool('transpose_a'), 'transpose_b', _op._get_attr_bool('transpose_b'), 'adjoint_a', _op._get_attr_bool('adjoint_a'), 'adjoint_b', _op._get_attr_bool('adjoint_b'), 'transpose_output', _op._get_attr_bool('transpose_output'), 'conjugate_output', _op._get_attr_bool('conjugate_output'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseMatrixMatMul', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result