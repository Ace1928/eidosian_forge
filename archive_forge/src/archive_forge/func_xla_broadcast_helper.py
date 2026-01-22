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
@tf_export('xla_broadcast_helper')
def xla_broadcast_helper(lhs: _atypes.TensorFuzzingAnnotation[TV_XlaBroadcastHelper_T], rhs: _atypes.TensorFuzzingAnnotation[TV_XlaBroadcastHelper_T], broadcast_dims: _atypes.TensorFuzzingAnnotation[TV_XlaBroadcastHelper_Tindices], name=None):
    """Helper operator for performing XLA-style broadcasts

  Broadcasts `lhs` and `rhs` to the same rank, by adding size 1 dimensions to
  whichever of `lhs` and `rhs` has the lower rank, using XLA's broadcasting rules
  for binary operators.

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the LHS input tensor
    rhs: A `Tensor`. Must have the same type as `lhs`. the RHS input tensor
    broadcast_dims: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      an XLA-style broadcast dimension specification
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (lhs_output, rhs_output).

    lhs_output: A `Tensor`. Has the same type as `lhs`. the broadcasted LHS tensor
    rhs_output: A `Tensor`. Has the same type as `lhs`. the broadcasted RHS tensor
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaBroadcastHelper', name, lhs, rhs, broadcast_dims)
            _result = _XlaBroadcastHelperOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_broadcast_helper((lhs, rhs, broadcast_dims, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_broadcast_helper_eager_fallback(lhs, rhs, broadcast_dims, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_broadcast_helper, (), dict(lhs=lhs, rhs=rhs, broadcast_dims=broadcast_dims, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_broadcast_helper((lhs, rhs, broadcast_dims, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaBroadcastHelper', lhs=lhs, rhs=rhs, broadcast_dims=broadcast_dims, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_broadcast_helper, (), dict(lhs=lhs, rhs=rhs, broadcast_dims=broadcast_dims, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaBroadcastHelper', _inputs_flat, _attrs, _result)
    _result = _XlaBroadcastHelperOutput._make(_result)
    return _result