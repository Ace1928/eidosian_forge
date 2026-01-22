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
@tf_export('rpc_get_value')
def rpc_get_value(status_or: _atypes.TensorFuzzingAnnotation[_atypes.Resource], Tout, name=None):
    """TODO: add doc.

  Args:
    status_or: A `Tensor` of type `resource`.
    Tout: A list of `tf.DTypes`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RpcGetValue', name, status_or, 'Tout', Tout)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_rpc_get_value((status_or, Tout, name), None)
            if _result is not NotImplemented:
                return _result
            return rpc_get_value_eager_fallback(status_or, Tout=Tout, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(rpc_get_value, (), dict(status_or=status_or, Tout=Tout, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_rpc_get_value((status_or, Tout, name), None)
        if _result is not NotImplemented:
            return _result
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'rpc_get_value' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('RpcGetValue', status_or=status_or, Tout=Tout, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(rpc_get_value, (), dict(status_or=status_or, Tout=Tout, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('Tout', _op.get_attr('Tout'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RpcGetValue', _inputs_flat, _attrs, _result)
    return _result