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
def matrix_logarithm(input: _atypes.TensorFuzzingAnnotation[TV_MatrixLogarithm_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_MatrixLogarithm_T]:
    """Computes the matrix logarithm of one or more square matrices:

  
  \\\\(log(exp(A)) = A\\\\)

  This op is only defined for complex matrices. If A is positive-definite and
  real, then casting to a complex matrix, taking the logarithm and casting back
  to a real matrix will give the correct result.

  This function computes the matrix logarithm using the Schur-Parlett algorithm.
  Details of the algorithm can be found in Section 11.6.2 of:
  Nicholas J. Higham, Functions of Matrices: Theory and Computation, SIAM 2008.
  ISBN 978-0-898716-46-7.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a tensor of the same shape as the input
  containing the exponential for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MatrixLogarithm', name, input)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return matrix_logarithm_eager_fallback(input, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MatrixLogarithm', input=input, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MatrixLogarithm', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result