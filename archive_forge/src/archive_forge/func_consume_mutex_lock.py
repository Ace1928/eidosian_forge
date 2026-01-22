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
def consume_mutex_lock(mutex_lock: _atypes.TensorFuzzingAnnotation[_atypes.Variant], name=None):
    """This op consumes a lock created by `MutexLock`.

  This op exists to consume a tensor created by `MutexLock` (other than
  direct control dependencies).  It should be the only that consumes the tensor,
  and will raise an error if it is not.  Its only purpose is to keep the
  mutex lock tensor alive until it is consumed by this op.

  **NOTE**: This operation must run on the same device as its input.  This may
  be enforced via the `colocate_with` mechanism.

  Args:
    mutex_lock: A `Tensor` of type `variant`.
      A tensor returned by `MutexLock`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ConsumeMutexLock', name, mutex_lock)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return consume_mutex_lock_eager_fallback(mutex_lock, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ConsumeMutexLock', mutex_lock=mutex_lock, name=name)
    return _op