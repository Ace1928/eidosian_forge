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
@tf_export('xla_dynamic_update_slice')
def xla_dynamic_update_slice(input: _atypes.TensorFuzzingAnnotation[TV_XlaDynamicUpdateSlice_T], update: _atypes.TensorFuzzingAnnotation[TV_XlaDynamicUpdateSlice_T], indices: _atypes.TensorFuzzingAnnotation[TV_XlaDynamicUpdateSlice_Tindices], name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaDynamicUpdateSlice_T]:
    """Wraps the XLA DynamicUpdateSlice operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#dynamicupdateslice
  .

  XlaDynamicUpdateSlice generates a result which is the value of the `input`
  operand, with a slice update overwritten at `indices`. The shape of `update`
  determines the shape of the sub-array of the result which is updated. The shape
  of indices must be rank == 1, with dimension size equal to the rank of `input`.

  Handling of out-of-bounds slice indices is implementation-defined.

  Args:
    input: A `Tensor`. A `Tensor` of type T.
    update: A `Tensor`. Must have the same type as `input`.
      A `Tensor` of type T. Same rank as `input`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into `input`. Must have length equal to the rank of
      `input`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. A `Tensor` of type T.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaDynamicUpdateSlice', name, input, update, indices)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_dynamic_update_slice((input, update, indices, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_dynamic_update_slice_eager_fallback(input, update, indices, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_dynamic_update_slice, (), dict(input=input, update=update, indices=indices, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_dynamic_update_slice((input, update, indices, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaDynamicUpdateSlice', input=input, update=update, indices=indices, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_dynamic_update_slice, (), dict(input=input, update=update, indices=indices, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaDynamicUpdateSlice', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result