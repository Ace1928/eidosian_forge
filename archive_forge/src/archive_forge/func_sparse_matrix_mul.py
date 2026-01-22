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
def sparse_matrix_mul(a: _atypes.TensorFuzzingAnnotation[_atypes.Variant], b: _atypes.TensorFuzzingAnnotation[TV_SparseMatrixMul_T], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Element-wise multiplication of a sparse matrix with a dense tensor.

  Returns a sparse matrix.

  The dense tensor `b` may be either a scalar; otherwise `a` must be a rank-3
  `SparseMatrix`; in this case `b` must be shaped `[batch_size, 1, 1]` and the
  multiply operation broadcasts.

  **NOTE** even if `b` is zero, the sparsity structure of the output does not
  change.

  Args:
    a: A `Tensor` of type `variant`. A CSRSparseMatrix.
    b: A `Tensor`. A dense tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseMatrixMul', name, a, b)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_matrix_mul_eager_fallback(a, b, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseMatrixMul', a=a, b=b, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseMatrixMul', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result