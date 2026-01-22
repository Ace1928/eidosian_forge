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
@tf_export('xla_reduce_window')
def xla_reduce_window(input: _atypes.TensorFuzzingAnnotation[TV_XlaReduceWindow_T], init_value: _atypes.TensorFuzzingAnnotation[TV_XlaReduceWindow_T], window_dimensions: _atypes.TensorFuzzingAnnotation[TV_XlaReduceWindow_Tindices], window_strides: _atypes.TensorFuzzingAnnotation[TV_XlaReduceWindow_Tindices], base_dilations: _atypes.TensorFuzzingAnnotation[TV_XlaReduceWindow_Tindices], window_dilations: _atypes.TensorFuzzingAnnotation[TV_XlaReduceWindow_Tindices], padding: _atypes.TensorFuzzingAnnotation[TV_XlaReduceWindow_Tindices], computation, name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaReduceWindow_T]:
    """Wraps the XLA ReduceWindow operator, documented at

   https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `bool`.
      the input tensor
    init_value: A `Tensor`. Must have the same type as `input`.
      a scalar representing the initial value for the reduction
    window_dimensions: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      the shape of the window
    window_strides: A `Tensor`. Must have the same type as `window_dimensions`.
      the inter-window strides
    base_dilations: A `Tensor`. Must have the same type as `window_dimensions`.
    window_dilations: A `Tensor`. Must have the same type as `window_dimensions`.
    padding: A `Tensor`. Must have the same type as `window_dimensions`.
      the padding to apply at the start and end of each input dimensions
    computation: A function decorated with @Defun. a reducer function to apply
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaReduceWindow', name, input, init_value, window_dimensions, window_strides, base_dilations, window_dilations, padding, 'computation', computation)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_reduce_window((input, init_value, window_dimensions, window_strides, base_dilations, window_dilations, padding, computation, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_reduce_window_eager_fallback(input, init_value, window_dimensions, window_strides, base_dilations, window_dilations, padding, computation=computation, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_reduce_window, (), dict(input=input, init_value=init_value, window_dimensions=window_dimensions, window_strides=window_strides, base_dilations=base_dilations, window_dilations=window_dilations, padding=padding, computation=computation, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_reduce_window((input, init_value, window_dimensions, window_strides, base_dilations, window_dilations, padding, computation, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaReduceWindow', input=input, init_value=init_value, window_dimensions=window_dimensions, window_strides=window_strides, base_dilations=base_dilations, window_dilations=window_dilations, padding=padding, computation=computation, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_reduce_window, (), dict(input=input, init_value=init_value, window_dimensions=window_dimensions, window_strides=window_strides, base_dilations=base_dilations, window_dilations=window_dilations, padding=padding, computation=computation, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'), 'computation', _op.get_attr('computation'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaReduceWindow', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result