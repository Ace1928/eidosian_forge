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
@tf_export('xla_custom_call')
def xla_custom_call(args, target_name: str, backend_config: str, dtype: TV_XlaCustomCall_dtype, shape, name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaCustomCall_dtype]:
    """Wraps the XLA CustomCall operator

    documented at https://www.tensorflow.org/xla/operation_semantics#customcall.

  Args:
    args: A list of `Tensor` objects.
      A list of `Tensor` with possibly different types.
    target_name: A `string`.
      Name of the function. A call instruction will be emitted which
      targets this symbol name.
    backend_config: A `string`.
      String, used to encode serialized metadata to the backend.
    dtype: A `tf.DType`. Output tensor data type.
    shape: A `tf.TensorShape` or list of `ints`. Output tensor shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaCustomCall', name, args, 'target_name', target_name, 'backend_config', backend_config, 'dtype', dtype, 'shape', shape)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_custom_call((args, target_name, backend_config, dtype, shape, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_custom_call_eager_fallback(args, target_name=target_name, backend_config=backend_config, dtype=dtype, shape=shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_custom_call, (), dict(args=args, target_name=target_name, backend_config=backend_config, dtype=dtype, shape=shape, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_custom_call((args, target_name, backend_config, dtype, shape, name), None)
        if _result is not NotImplemented:
            return _result
    target_name = _execute.make_str(target_name, 'target_name')
    backend_config = _execute.make_str(backend_config, 'backend_config')
    dtype = _execute.make_type(dtype, 'dtype')
    shape = _execute.make_shape(shape, 'shape')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaCustomCall', args=args, target_name=target_name, backend_config=backend_config, dtype=dtype, shape=shape, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_custom_call, (), dict(args=args, target_name=target_name, backend_config=backend_config, dtype=dtype, shape=shape, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('target_name', _op.get_attr('target_name'), 'backend_config', _op.get_attr('backend_config'), 'T', _op.get_attr('T'), 'dtype', _op._get_attr_type('dtype'), 'shape', _op.get_attr('shape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaCustomCall', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result