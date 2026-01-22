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
def sparse_add_grad(backprop_val_grad: _atypes.TensorFuzzingAnnotation[TV_SparseAddGrad_T], a_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], b_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], sum_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name=None):
    """The gradient operator for the SparseAdd op.

  The SparseAdd op calculates A + B, where A, B, and the sum are all represented
  as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
  non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
  values of A and B.

  Args:
    backprop_val_grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D with shape `[nnz(sum)]`.  The gradient with respect to
      the non-empty values of the sum.
    a_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the `SparseTensor` A, size `[nnz(A), ndims]`.
    b_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the `SparseTensor` B, size `[nnz(B), ndims]`.
    sum_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the sum `SparseTensor`, size
      `[nnz(sum), ndims]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a_val_grad, b_val_grad).

    a_val_grad: A `Tensor`. Has the same type as `backprop_val_grad`.
    b_val_grad: A `Tensor`. Has the same type as `backprop_val_grad`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseAddGrad', name, backprop_val_grad, a_indices, b_indices, sum_indices)
            _result = _SparseAddGradOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_add_grad_eager_fallback(backprop_val_grad, a_indices, b_indices, sum_indices, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseAddGrad', backprop_val_grad=backprop_val_grad, a_indices=a_indices, b_indices=b_indices, sum_indices=sum_indices, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseAddGrad', _inputs_flat, _attrs, _result)
    _result = _SparseAddGradOutput._make(_result)
    return _result