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
@tf_export('xla_dequantize')
def xla_dequantize(input: _atypes.TensorFuzzingAnnotation[_atypes.UInt32], min_range: float, max_range: float, mode: str, transpose_output: bool, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.BFloat16]:
    """Takes the packed uint32 input and unpacks the input to uint8 to do

  Dequantization on device.

  Args:
    input: A `Tensor` of type `uint32`.
      Input tensors whose types is uint32, shape is [d0, ..., dn].
    min_range: A `float`.
      The minimum scalar value possibly produced for the input.
    max_range: A `float`.
      The maximum scalar value possibly produced for the input.
    mode: A `string`.
      String to determine the dequantize mode in {"MIN_COMBINED", "MIN_FIRST", "SCALED"}.
    transpose_output: A `bool`.
      Boolean to determine if output is transposed. transpose_output
      is faster when input is large and rank of input is higher than 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bfloat16`.
    Output tensors whose types is bfloat16. If transpose_output is true,
    output shape is [dn * 4, dn-1, ..., d1, d0]. If transpose_output
    is false, output shape is [d0,..., dn * 4].
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaDequantize', name, input, 'min_range', min_range, 'max_range', max_range, 'mode', mode, 'transpose_output', transpose_output)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_dequantize((input, min_range, max_range, mode, transpose_output, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_dequantize_eager_fallback(input, min_range=min_range, max_range=max_range, mode=mode, transpose_output=transpose_output, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_dequantize, (), dict(input=input, min_range=min_range, max_range=max_range, mode=mode, transpose_output=transpose_output, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_dequantize((input, min_range, max_range, mode, transpose_output, name), None)
        if _result is not NotImplemented:
            return _result
    min_range = _execute.make_float(min_range, 'min_range')
    max_range = _execute.make_float(max_range, 'max_range')
    mode = _execute.make_str(mode, 'mode')
    transpose_output = _execute.make_bool(transpose_output, 'transpose_output')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaDequantize', input=input, min_range=min_range, max_range=max_range, mode=mode, transpose_output=transpose_output, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_dequantize, (), dict(input=input, min_range=min_range, max_range=max_range, mode=mode, transpose_output=transpose_output, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('min_range', _op.get_attr('min_range'), 'max_range', _op.get_attr('max_range'), 'mode', _op.get_attr('mode'), 'transpose_output', _op._get_attr_bool('transpose_output'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaDequantize', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result