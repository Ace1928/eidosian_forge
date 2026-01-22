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
@tf_export('dropout_grad')
def dropout_grad(gradients: _atypes.TensorFuzzingAnnotation[TV_DropoutGrad_T], rate: _atypes.TensorFuzzingAnnotation[TV_DropoutGrad_T], mask: _atypes.TensorFuzzingAnnotation[_atypes.UInt8], name=None) -> _atypes.TensorFuzzingAnnotation[TV_DropoutGrad_T]:
    """Computes gradients of the dropout function.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `half`, `float64`.
    rate: A `Tensor`. Must have the same type as `gradients`.
    mask: A `Tensor` of type `uint8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DropoutGrad', name, gradients, rate, mask)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_dropout_grad((gradients, rate, mask, name), None)
            if _result is not NotImplemented:
                return _result
            return dropout_grad_eager_fallback(gradients, rate, mask, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(dropout_grad, (), dict(gradients=gradients, rate=rate, mask=mask, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_dropout_grad((gradients, rate, mask, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('DropoutGrad', gradients=gradients, rate=rate, mask=mask, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(dropout_grad, (), dict(gradients=gradients, rate=rate, mask=mask, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DropoutGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result