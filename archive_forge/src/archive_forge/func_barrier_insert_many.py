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
def barrier_insert_many(handle: _atypes.TensorFuzzingAnnotation[_atypes.String], keys: _atypes.TensorFuzzingAnnotation[_atypes.String], values: _atypes.TensorFuzzingAnnotation[TV_BarrierInsertMany_T], component_index: int, name=None):
    """For each key, assigns the respective value to the specified component.

  If a key is not found in the barrier, this operation will create a new
  incomplete element. If a key is found in the barrier, and the element
  already has a value at component_index, this operation will fail with
  INVALID_ARGUMENT, and leave the barrier in an undefined state.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    keys: A `Tensor` of type `string`.
      A one-dimensional tensor of keys, with length n.
    values: A `Tensor`.
      An any-dimensional tensor of values, which are associated with the
      respective keys. The 0th dimension must have length n.
    component_index: An `int`.
      The component of the barrier elements that is being assigned.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("barrier_insert_many op does not support eager execution. Arg 'handle' is a ref.")
    component_index = _execute.make_int(component_index, 'component_index')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BarrierInsertMany', handle=handle, keys=keys, values=values, component_index=component_index, name=name)
    return _op