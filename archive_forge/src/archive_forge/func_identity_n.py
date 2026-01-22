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
@tf_export('identity_n')
def identity_n(input, name=None):
    """Returns a list of tensors with the same shapes and contents as the input

  tensors.

  This op can be used to override the gradient for complicated functions. For
  example, suppose y = f(x) and we wish to apply a custom function g for backprop
  such that dx = g(dy). In Python,

  ```python
  with tf.get_default_graph().gradient_override_map(
      {'IdentityN': 'OverrideGradientWithG'}):
    y, _ = identity_n([f(x), x])

  @tf.RegisterGradient('OverrideGradientWithG')
  def ApplyG(op, dy, _):
    return [None, g(dy)]  # Do not backprop to f(x).
  ```

  Args:
    input: A list of `Tensor` objects.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'IdentityN', name, input)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_identity_n((input, name), None)
            if _result is not NotImplemented:
                return _result
            return identity_n_eager_fallback(input, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(identity_n, (), dict(input=input, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_identity_n((input, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('IdentityN', input=input, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(identity_n, (), dict(input=input, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op.get_attr('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('IdentityN', _inputs_flat, _attrs, _result)
    return _result