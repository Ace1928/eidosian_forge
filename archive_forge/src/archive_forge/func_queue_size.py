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
def queue_size(handle: _atypes.TensorFuzzingAnnotation[_atypes.String], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int32]:
    """Computes the number of elements in the given queue.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("queue_size op does not support eager execution. Arg 'handle' is a ref.")
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QueueSize', handle=handle, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('QueueSize', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result