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
@tf_export('xla_custom_call_v2')
def xla_custom_call_v2(operands, call_target_name: str, backend_config: str, has_side_effect: bool, result_dtypes, result_shapes, name=None):
    """Emits an HLO `CustomCall` operation with multiple outputs.

  As opposed to `XlaCustomCall`, this operation supports multiple outputs.

  See `CustomCall` specification at
    https://tensorflow.org/xla/operation_semantics#customcall,
  and `mhlo.custom_call` specification at
    https://tensorflow.org/mlir/hlo_ops#mhlocustom_call_mlirmhlocustomcallop.

  Args:
    operands: A list of `Tensor` objects.
      A sequence of tensors with possibly different types.
    call_target_name: A `string`.
      Name of the user function. The function signature must conform
      to version 3 of the API, see `API_VERSION_STATUS_RETURNING_UNIFIED`. All
      operands and results assumed to be in the default layout.
    backend_config: A `string`.
      A string that encodes a metadata for the backend.
    has_side_effect: A `bool`.
      Indicates whether the custom call has side effects.
    result_dtypes: A list of `tf.DTypes`. Types of all results.
    result_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      Shapes of all results.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `result_dtypes`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaCustomCallV2', name, operands, 'call_target_name', call_target_name, 'backend_config', backend_config, 'has_side_effect', has_side_effect, 'result_dtypes', result_dtypes, 'result_shapes', result_shapes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_custom_call_v2((operands, call_target_name, backend_config, has_side_effect, result_dtypes, result_shapes, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_custom_call_v2_eager_fallback(operands, call_target_name=call_target_name, backend_config=backend_config, has_side_effect=has_side_effect, result_dtypes=result_dtypes, result_shapes=result_shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_custom_call_v2, (), dict(operands=operands, call_target_name=call_target_name, backend_config=backend_config, has_side_effect=has_side_effect, result_dtypes=result_dtypes, result_shapes=result_shapes, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_custom_call_v2((operands, call_target_name, backend_config, has_side_effect, result_dtypes, result_shapes, name), None)
        if _result is not NotImplemented:
            return _result
    call_target_name = _execute.make_str(call_target_name, 'call_target_name')
    backend_config = _execute.make_str(backend_config, 'backend_config')
    has_side_effect = _execute.make_bool(has_side_effect, 'has_side_effect')
    if not isinstance(result_dtypes, (list, tuple)):
        raise TypeError("Expected list for 'result_dtypes' argument to 'xla_custom_call_v2' Op, not %r." % result_dtypes)
    result_dtypes = [_execute.make_type(_t, 'result_dtypes') for _t in result_dtypes]
    if not isinstance(result_shapes, (list, tuple)):
        raise TypeError("Expected list for 'result_shapes' argument to 'xla_custom_call_v2' Op, not %r." % result_shapes)
    result_shapes = [_execute.make_shape(_s, 'result_shapes') for _s in result_shapes]
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaCustomCallV2', operands=operands, call_target_name=call_target_name, backend_config=backend_config, has_side_effect=has_side_effect, result_dtypes=result_dtypes, result_shapes=result_shapes, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_custom_call_v2, (), dict(operands=operands, call_target_name=call_target_name, backend_config=backend_config, has_side_effect=has_side_effect, result_dtypes=result_dtypes, result_shapes=result_shapes, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('call_target_name', _op.get_attr('call_target_name'), 'backend_config', _op.get_attr('backend_config'), 'has_side_effect', _op._get_attr_bool('has_side_effect'), 'operand_dtypes', _op.get_attr('operand_dtypes'), 'result_dtypes', _op.get_attr('result_dtypes'), 'result_shapes', _op.get_attr('result_shapes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaCustomCallV2', _inputs_flat, _attrs, _result)
    return _result