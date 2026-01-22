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
@tf_export('n_polymorphic_restrict_out')
def n_polymorphic_restrict_out(T: TV_NPolymorphicRestrictOut_T, N: int, name=None):
    """TODO: add doc.

  Args:
    T: A `tf.DType` from: `tf.string, tf.bool`.
    N: An `int` that is `>= 2`.
    name: A name for the operation (optional).

  Returns:
    A list of `N` `Tensor` objects with type `T`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'NPolymorphicRestrictOut', name, 'T', T, 'N', N)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_n_polymorphic_restrict_out((T, N, name), None)
            if _result is not NotImplemented:
                return _result
            return n_polymorphic_restrict_out_eager_fallback(T=T, N=N, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(n_polymorphic_restrict_out, (), dict(T=T, N=N, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_n_polymorphic_restrict_out((T, N, name), None)
        if _result is not NotImplemented:
            return _result
    T = _execute.make_type(T, 'T')
    N = _execute.make_int(N, 'N')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('NPolymorphicRestrictOut', T=T, N=N, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(n_polymorphic_restrict_out, (), dict(T=T, N=N, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'N', _op._get_attr_int('N'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('NPolymorphicRestrictOut', _inputs_flat, _attrs, _result)
    return _result