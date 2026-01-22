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
def sparse_slice_grad(backprop_val_grad: _atypes.TensorFuzzingAnnotation[TV_SparseSliceGrad_T], input_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], input_start: _atypes.TensorFuzzingAnnotation[_atypes.Int64], output_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name=None) -> _atypes.TensorFuzzingAnnotation[TV_SparseSliceGrad_T]:
    """The gradient operator for the SparseSlice op.

  This op takes in the upstream gradient w.r.t. non-empty values of
  the sliced `SparseTensor`, and outputs the gradients w.r.t.
  the non-empty values of input `SparseTensor`.

  Args:
    backprop_val_grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D. The gradient with respect to
      the non-empty values of the sliced `SparseTensor`.
    input_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the input `SparseTensor`.
    input_start: A `Tensor` of type `int64`.
      1-D. tensor represents the start of the slice.
    output_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the sliced `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `backprop_val_grad`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseSliceGrad', name, backprop_val_grad, input_indices, input_start, output_indices)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_slice_grad_eager_fallback(backprop_val_grad, input_indices, input_start, output_indices, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseSliceGrad', backprop_val_grad=backprop_val_grad, input_indices=input_indices, input_start=input_start, output_indices=output_indices, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseSliceGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result