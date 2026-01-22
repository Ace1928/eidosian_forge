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
@tf_export('strings.lower')
def string_lower(input: _atypes.TensorFuzzingAnnotation[_atypes.String], encoding: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Converts all uppercase characters into their respective lowercase replacements.

  Example:

  >>> tf.strings.lower("CamelCase string and ALL CAPS")
  <tf.Tensor: shape=(), dtype=string, numpy=b'camelcase string and all caps'>

  Args:
    input: A `Tensor` of type `string`. The input to be lower-cased.
    encoding: An optional `string`. Defaults to `""`.
      Character encoding of `input`. Allowed values are '' and 'utf-8'.
      Value '' is interpreted as ASCII.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StringLower', name, input, 'encoding', encoding)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_string_lower((input, encoding, name), None)
            if _result is not NotImplemented:
                return _result
            return string_lower_eager_fallback(input, encoding=encoding, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(string_lower, (), dict(input=input, encoding=encoding, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_string_lower((input, encoding, name), None)
        if _result is not NotImplemented:
            return _result
    if encoding is None:
        encoding = ''
    encoding = _execute.make_str(encoding, 'encoding')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('StringLower', input=input, encoding=encoding, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(string_lower, (), dict(input=input, encoding=encoding, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('encoding', _op.get_attr('encoding'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StringLower', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result