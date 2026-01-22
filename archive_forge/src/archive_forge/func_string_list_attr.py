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
@tf_export('string_list_attr')
def string_list_attr(a, b: str, name=None):
    """TODO: add doc.

  Args:
    a: A list of `strings`.
    b: A `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StringListAttr', name, 'a', a, 'b', b)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_string_list_attr((a, b, name), None)
            if _result is not NotImplemented:
                return _result
            return string_list_attr_eager_fallback(a=a, b=b, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(string_list_attr, (), dict(a=a, b=b, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_string_list_attr((a, b, name), None)
        if _result is not NotImplemented:
            return _result
    if not isinstance(a, (list, tuple)):
        raise TypeError("Expected list for 'a' argument to 'string_list_attr' Op, not %r." % a)
    a = [_execute.make_str(_s, 'a') for _s in a]
    b = _execute.make_str(b, 'b')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('StringListAttr', a=a, b=b, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(string_list_attr, (), dict(a=a, b=b, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    return _op