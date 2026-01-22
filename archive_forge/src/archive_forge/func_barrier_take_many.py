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
def barrier_take_many(handle: _atypes.TensorFuzzingAnnotation[_atypes.String], num_elements: _atypes.TensorFuzzingAnnotation[_atypes.Int32], component_types, allow_small_batch: bool=False, wait_for_incomplete: bool=False, timeout_ms: int=-1, name=None):
    """Takes the given number of completed elements from a barrier.

  This operation concatenates completed-element component tensors along
  the 0th dimension to make a single component tensor.

  Elements come out of the barrier when they are complete, and in the order
  in which they were placed into the barrier.  The indices output provides
  information about the batch in which each element was originally inserted
  into the barrier.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    num_elements: A `Tensor` of type `int32`.
      A single-element tensor containing the number of elements to
      take.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    allow_small_batch: An optional `bool`. Defaults to `False`.
      Allow to return less than num_elements items if barrier is
      already closed.
    wait_for_incomplete: An optional `bool`. Defaults to `False`.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is empty, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, keys, values).

    indices: A `Tensor` of type `int64`.
    keys: A `Tensor` of type `string`.
    values: A list of `Tensor` objects of type `component_types`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("barrier_take_many op does not support eager execution. Arg 'handle' is a ref.")
    if not isinstance(component_types, (list, tuple)):
        raise TypeError("Expected list for 'component_types' argument to 'barrier_take_many' Op, not %r." % component_types)
    component_types = [_execute.make_type(_t, 'component_types') for _t in component_types]
    if allow_small_batch is None:
        allow_small_batch = False
    allow_small_batch = _execute.make_bool(allow_small_batch, 'allow_small_batch')
    if wait_for_incomplete is None:
        wait_for_incomplete = False
    wait_for_incomplete = _execute.make_bool(wait_for_incomplete, 'wait_for_incomplete')
    if timeout_ms is None:
        timeout_ms = -1
    timeout_ms = _execute.make_int(timeout_ms, 'timeout_ms')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BarrierTakeMany', handle=handle, num_elements=num_elements, component_types=component_types, allow_small_batch=allow_small_batch, wait_for_incomplete=wait_for_incomplete, timeout_ms=timeout_ms, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('component_types', _op.get_attr('component_types'), 'allow_small_batch', _op._get_attr_bool('allow_small_batch'), 'wait_for_incomplete', _op._get_attr_bool('wait_for_incomplete'), 'timeout_ms', _op._get_attr_int('timeout_ms'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BarrierTakeMany', _inputs_flat, _attrs, _result)
    _result = _result[:2] + [_result[2:]]
    _result = _BarrierTakeManyOutput._make(_result)
    return _result