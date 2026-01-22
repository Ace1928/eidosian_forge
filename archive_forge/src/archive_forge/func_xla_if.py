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
@tf_export('xla_if')
def xla_if(cond: _atypes.TensorFuzzingAnnotation[TV_XlaIf_Tcond], inputs, then_branch, else_branch, Tout, name=None):
    """output = cond ? then_branch(inputs) : else_branch(inputs).

  Args:
    cond: A `Tensor`. A boolean scalar.
    inputs: A list of `Tensor` objects. A list of input tensors.
    then_branch: A function decorated with @Defun.
      A function takes 'inputs' and returns a list of tensors,
      whose types are the same as what else_branch returns.
    else_branch: A function decorated with @Defun.
      A function takes 'inputs' and returns a list of tensors.
      whose types are the same as what then_branch returns.
    Tout: A list of `tf.DTypes`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
    A list of tensors returned by either then_branch(inputs) or
    else_branch(inputs). The input shapes of the then_branch and
    else_branch must match.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaIf', name, cond, inputs, 'then_branch', then_branch, 'else_branch', else_branch, 'Tout', Tout)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_if((cond, inputs, then_branch, else_branch, Tout, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_if_eager_fallback(cond, inputs, then_branch=then_branch, else_branch=else_branch, Tout=Tout, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_if, (), dict(cond=cond, inputs=inputs, then_branch=then_branch, else_branch=else_branch, Tout=Tout, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_if((cond, inputs, then_branch, else_branch, Tout, name), None)
        if _result is not NotImplemented:
            return _result
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'xla_if' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaIf', cond=cond, inputs=inputs, then_branch=then_branch, else_branch=else_branch, Tout=Tout, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_if, (), dict(cond=cond, inputs=inputs, then_branch=then_branch, else_branch=else_branch, Tout=Tout, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('Tcond', _op._get_attr_type('Tcond'), 'then_branch', _op.get_attr('then_branch'), 'else_branch', _op.get_attr('else_branch'), 'Tin', _op.get_attr('Tin'), 'Tout', _op.get_attr('Tout'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaIf', _inputs_flat, _attrs, _result)
    return _result