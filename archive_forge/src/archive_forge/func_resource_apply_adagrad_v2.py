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
def resource_apply_adagrad_v2(var: _atypes.TensorFuzzingAnnotation[_atypes.Resource], accum: _atypes.TensorFuzzingAnnotation[_atypes.Resource], lr: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyAdagradV2_T], epsilon: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyAdagradV2_T], grad: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyAdagradV2_T], use_locking: bool=False, update_slots: bool=True, name=None):
    """Update '*var' according to the adagrad scheme.

  accum += grad * grad
  var -= lr * grad * (1 / (sqrt(accum) + epsilon))

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    update_slots: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResourceApplyAdagradV2', name, var, accum, lr, epsilon, grad, 'use_locking', use_locking, 'update_slots', update_slots)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resource_apply_adagrad_v2_eager_fallback(var, accum, lr, epsilon, grad, use_locking=use_locking, update_slots=update_slots, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if use_locking is None:
        use_locking = False
    use_locking = _execute.make_bool(use_locking, 'use_locking')
    if update_slots is None:
        update_slots = True
    update_slots = _execute.make_bool(update_slots, 'update_slots')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResourceApplyAdagradV2', var=var, accum=accum, lr=lr, epsilon=epsilon, grad=grad, use_locking=use_locking, update_slots=update_slots, name=name)
    return _op