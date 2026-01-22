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
def ordered_map_stage(key: _atypes.TensorFuzzingAnnotation[_atypes.Int64], indices: _atypes.TensorFuzzingAnnotation[_atypes.Int32], values, dtypes, capacity: int=0, memory_limit: int=0, container: str='', shared_name: str='', name=None):
    """Stage (key, values) in the underlying container which behaves like a ordered

  associative container.   Elements are ordered by key.

  Args:
    key: A `Tensor` of type `int64`. int64
    indices: A `Tensor` of type `int32`.
    values: A list of `Tensor` objects. a list of tensors
      dtypes A list of data types that inserted values should adhere to.
    dtypes: A list of `tf.DTypes`.
    capacity: An optional `int` that is `>= 0`. Defaults to `0`.
      Maximum number of elements in the Staging Area. If > 0, inserts
      on the container will block when the capacity is reached.
    memory_limit: An optional `int` that is `>= 0`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container. Otherwise,
      a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      It is necessary to match this name to the matching Unstage Op.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'OrderedMapStage', name, key, indices, values, 'capacity', capacity, 'memory_limit', memory_limit, 'dtypes', dtypes, 'container', container, 'shared_name', shared_name)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return ordered_map_stage_eager_fallback(key, indices, values, capacity=capacity, memory_limit=memory_limit, dtypes=dtypes, container=container, shared_name=shared_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(dtypes, (list, tuple)):
        raise TypeError("Expected list for 'dtypes' argument to 'ordered_map_stage' Op, not %r." % dtypes)
    dtypes = [_execute.make_type(_t, 'dtypes') for _t in dtypes]
    if capacity is None:
        capacity = 0
    capacity = _execute.make_int(capacity, 'capacity')
    if memory_limit is None:
        memory_limit = 0
    memory_limit = _execute.make_int(memory_limit, 'memory_limit')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('OrderedMapStage', key=key, indices=indices, values=values, dtypes=dtypes, capacity=capacity, memory_limit=memory_limit, container=container, shared_name=shared_name, name=name)
    return _op