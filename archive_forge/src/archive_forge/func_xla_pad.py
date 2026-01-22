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
@tf_export('xla_pad')
def xla_pad(input: _atypes.TensorFuzzingAnnotation[TV_XlaPad_T], padding_value: _atypes.TensorFuzzingAnnotation[TV_XlaPad_T], padding_low: _atypes.TensorFuzzingAnnotation[TV_XlaPad_Tindices], padding_high: _atypes.TensorFuzzingAnnotation[TV_XlaPad_Tindices], padding_interior: _atypes.TensorFuzzingAnnotation[TV_XlaPad_Tindices], name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaPad_T]:
    """Wraps the XLA Pad operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#pad
  .

  Args:
    input: A `Tensor`. A `Tensor` of type T.
    padding_value: A `Tensor`. Must have the same type as `input`.
      A scalar `Tensor` of type T.
    padding_low: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      the padding to apply at the start of each input dimensions. Must
      be a compile-time constant 1D tensor of length equal to rank of input.
    padding_high: A `Tensor`. Must have the same type as `padding_low`.
      the padding to apply at the end of each input dimension. Must
      be a compile-time constant 1D tensor of length equal to rank of input.
    padding_interior: A `Tensor`. Must have the same type as `padding_low`.
      the padding to apply between each input element. Must
      be a compile-time constant 1D tensor of length equal to rank of input,
      containing only non-negative values.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. A `Tensor` of type T.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaPad', name, input, padding_value, padding_low, padding_high, padding_interior)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_pad((input, padding_value, padding_low, padding_high, padding_interior, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_pad_eager_fallback(input, padding_value, padding_low, padding_high, padding_interior, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_pad, (), dict(input=input, padding_value=padding_value, padding_low=padding_low, padding_high=padding_high, padding_interior=padding_interior, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_pad((input, padding_value, padding_low, padding_high, padding_interior, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaPad', input=input, padding_value=padding_value, padding_low=padding_low, padding_high=padding_high, padding_interior=padding_interior, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_pad, (), dict(input=input, padding_value=padding_value, padding_low=padding_low, padding_high=padding_high, padding_interior=padding_interior, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaPad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result