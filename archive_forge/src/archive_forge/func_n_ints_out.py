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
@tf_export('n_ints_out')
def n_ints_out(N: int, name=None):
    """TODO: add doc.

  Args:
    N: An `int` that is `>= 2`.
    name: A name for the operation (optional).

  Returns:
    A list of `N` `Tensor` objects with type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'NIntsOut', name, 'N', N)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_n_ints_out((N, name), None)
            if _result is not NotImplemented:
                return _result
            return n_ints_out_eager_fallback(N=N, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(n_ints_out, (), dict(N=N, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_n_ints_out((N, name), None)
        if _result is not NotImplemented:
            return _result
    N = _execute.make_int(N, 'N')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('NIntsOut', N=N, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(n_ints_out, (), dict(N=N, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('N', _op._get_attr_int('N'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('NIntsOut', _inputs_flat, _attrs, _result)
    return _result