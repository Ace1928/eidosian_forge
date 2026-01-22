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
def delete_multi_device_iterator(multi_device_iterator: _atypes.TensorFuzzingAnnotation[_atypes.Resource], iterators: List[_atypes.TensorFuzzingAnnotation[_atypes.Resource]], deleter: _atypes.TensorFuzzingAnnotation[_atypes.Variant], name=None):
    """A container for an iterator resource.

  Args:
    multi_device_iterator: A `Tensor` of type `resource`.
      A handle to the multi device iterator to delete.
    iterators: A list of `Tensor` objects with type `resource`.
      A list of iterator handles (unused). This is added so that automatic control dependencies get added during function tracing that ensure this op runs after all the dependent iterators are deleted.
    deleter: A `Tensor` of type `variant`. A variant deleter.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DeleteMultiDeviceIterator', name, multi_device_iterator, iterators, deleter)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return delete_multi_device_iterator_eager_fallback(multi_device_iterator, iterators, deleter, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(iterators, (list, tuple)):
        raise TypeError("Expected list for 'iterators' argument to 'delete_multi_device_iterator' Op, not %r." % iterators)
    _attr_N = len(iterators)
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DeleteMultiDeviceIterator', multi_device_iterator=multi_device_iterator, iterators=iterators, deleter=deleter, name=name)
    return _op