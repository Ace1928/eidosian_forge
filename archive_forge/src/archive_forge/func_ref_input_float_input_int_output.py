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
@tf_export('ref_input_float_input_int_output')
def ref_input_float_input_int_output(a: _atypes.TensorFuzzingAnnotation[_atypes.Float32], b: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int32]:
    """TODO: add doc.

  Args:
    a: A `Tensor` of type mutable `float32`.
    b: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("ref_input_float_input_int_output op does not support eager execution. Arg 'a' is a ref.")
    else:
        _result = _dispatcher_for_ref_input_float_input_int_output((a, b, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('RefInputFloatInputIntOutput', a=a, b=b, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(ref_input_float_input_int_output, (), dict(a=a, b=b, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('RefInputFloatInputIntOutput', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result