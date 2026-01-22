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
def sparse_fill_empty_rows_grad(reverse_index_map: _atypes.TensorFuzzingAnnotation[_atypes.Int64], grad_values: _atypes.TensorFuzzingAnnotation[TV_SparseFillEmptyRowsGrad_T], name=None):
    """The gradient of SparseFillEmptyRows.

  Takes vectors reverse_index_map, shaped `[N]`, and grad_values,
  shaped `[N_full]`, where `N_full >= N` and copies data into either
  `d_values` or `d_default_value`.  Here `d_values` is shaped `[N]` and
  `d_default_value` is a scalar.

    d_values[j] = grad_values[reverse_index_map[j]]
    d_default_value = sum_{k : 0 .. N_full - 1} (
       grad_values[k] * 1{k not in reverse_index_map})

  Args:
    reverse_index_map: A `Tensor` of type `int64`.
      1-D.  The reverse index map from SparseFillEmptyRows.
    grad_values: A `Tensor`. 1-D.  The gradients from backprop.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (d_values, d_default_value).

    d_values: A `Tensor`. Has the same type as `grad_values`.
    d_default_value: A `Tensor`. Has the same type as `grad_values`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseFillEmptyRowsGrad', name, reverse_index_map, grad_values)
            _result = _SparseFillEmptyRowsGradOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_fill_empty_rows_grad_eager_fallback(reverse_index_map, grad_values, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseFillEmptyRowsGrad', reverse_index_map=reverse_index_map, grad_values=grad_values, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseFillEmptyRowsGrad', _inputs_flat, _attrs, _result)
    _result = _SparseFillEmptyRowsGradOutput._make(_result)
    return _result