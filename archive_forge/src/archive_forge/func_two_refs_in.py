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
@tf_export('two_refs_in')
def two_refs_in(a: _atypes.TensorFuzzingAnnotation[TV_TwoRefsIn_T], b: _atypes.TensorFuzzingAnnotation[TV_TwoRefsIn_T], name=None):
    """TODO: add doc.

  Args:
    a: A mutable `Tensor`.
    b: A mutable `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("two_refs_in op does not support eager execution. Arg 'b' is a ref.")
    else:
        _result = _dispatcher_for_two_refs_in((a, b, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('TwoRefsIn', a=a, b=b, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(two_refs_in, (), dict(a=a, b=b, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    return _op