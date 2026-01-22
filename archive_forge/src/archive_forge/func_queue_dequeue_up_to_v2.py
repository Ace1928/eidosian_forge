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
def queue_dequeue_up_to_v2(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], n: _atypes.TensorFuzzingAnnotation[_atypes.Int32], component_types, timeout_ms: int=-1, name=None):
    """Dequeues `n` tuples of one or more tensors from the given queue.

  This operation is not supported by all queues.  If a queue does not support
  DequeueUpTo, then an Unimplemented error is returned.

  If the queue is closed and there are more than 0 but less than `n`
  elements remaining, then instead of returning an OutOfRange error like
  QueueDequeueMany, less than `n` elements are returned immediately.  If
  the queue is closed and there are 0 elements left in the queue, then
  an OutOfRange error is returned just like in QueueDequeueMany.
  Otherwise the behavior is identical to QueueDequeueMany:

  This operation concatenates queue-element component tensors along the
  0th dimension to make a single component tensor.  All of the components
  in the dequeued tuple will have size n in the 0th dimension.

  This operation has `k` outputs, where `k` is the number of components in
  the tuples stored in the given queue, and output `i` is the ith
  component of the dequeued tuple.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    n: A `Tensor` of type `int32`. The number of tuples to dequeue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue has fewer than n elements, this operation
      will block for up to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QueueDequeueUpToV2', name, handle, n, 'component_types', component_types, 'timeout_ms', timeout_ms)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return queue_dequeue_up_to_v2_eager_fallback(handle, n, component_types=component_types, timeout_ms=timeout_ms, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(component_types, (list, tuple)):
        raise TypeError("Expected list for 'component_types' argument to 'queue_dequeue_up_to_v2' Op, not %r." % component_types)
    component_types = [_execute.make_type(_t, 'component_types') for _t in component_types]
    if timeout_ms is None:
        timeout_ms = -1
    timeout_ms = _execute.make_int(timeout_ms, 'timeout_ms')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QueueDequeueUpToV2', handle=handle, n=n, component_types=component_types, timeout_ms=timeout_ms, name=name)
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('component_types', _op.get_attr('component_types'), 'timeout_ms', _op._get_attr_int('timeout_ms'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QueueDequeueUpToV2', _inputs_flat, _attrs, _result)
    return _result