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
def resource_count_up_to(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], limit: int, T: TV_ResourceCountUpTo_T, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ResourceCountUpTo_T]:
    """Increments variable pointed to by 'resource' until it reaches 'limit'.

  Args:
    resource: A `Tensor` of type `resource`.
      Should be from a scalar `Variable` node.
    limit: An `int`.
      If incrementing ref would bring it above limit, instead generates an
      'OutOfRange' error.
    T: A `tf.DType` from: `tf.int32, tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `T`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResourceCountUpTo', name, resource, 'limit', limit, 'T', T)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resource_count_up_to_eager_fallback(resource, limit=limit, T=T, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    limit = _execute.make_int(limit, 'limit')
    T = _execute.make_type(T, 'T')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResourceCountUpTo', resource=resource, limit=limit, T=T, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('limit', _op._get_attr_int('limit'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ResourceCountUpTo', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result