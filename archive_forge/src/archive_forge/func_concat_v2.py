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
def concat_v2(values: List[_atypes.TensorFuzzingAnnotation[TV_ConcatV2_T]], axis: _atypes.TensorFuzzingAnnotation[TV_ConcatV2_Tidx], name=None) -> _atypes.TensorFuzzingAnnotation[TV_ConcatV2_T]:
    """Concatenates tensors along one dimension.

  Args:
    values: A list of at least 2 `Tensor` objects with the same type.
      List of `N` Tensors to concatenate. Their ranks and types must match,
      and their sizes must match in all dimensions except `concat_dim`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D.  The dimension along which to concatenate.  Must be in the
      range [-rank(values), rank(values)).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `values`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ConcatV2', name, values, axis)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return concat_v2_eager_fallback(values, axis, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(values, (list, tuple)):
        raise TypeError("Expected list for 'values' argument to 'concat_v2' Op, not %r." % values)
    _attr_N = len(values)
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ConcatV2', values=values, axis=axis, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('N', _op._get_attr_int('N'), 'T', _op._get_attr_type('T'), 'Tidx', _op._get_attr_type('Tidx'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ConcatV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result