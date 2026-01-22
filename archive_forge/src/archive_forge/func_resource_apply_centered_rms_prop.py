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
def resource_apply_centered_rms_prop(var: _atypes.TensorFuzzingAnnotation[_atypes.Resource], mg: _atypes.TensorFuzzingAnnotation[_atypes.Resource], ms: _atypes.TensorFuzzingAnnotation[_atypes.Resource], mom: _atypes.TensorFuzzingAnnotation[_atypes.Resource], lr: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyCenteredRMSProp_T], rho: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyCenteredRMSProp_T], momentum: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyCenteredRMSProp_T], epsilon: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyCenteredRMSProp_T], grad: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyCenteredRMSProp_T], use_locking: bool=False, name=None):
    """Update '*var' according to the centered RMSProp algorithm.

  The centered RMSProp algorithm uses an estimate of the centered second moment
  (i.e., the variance) for normalization, as opposed to regular RMSProp, which
  uses the (uncentered) second moment. This often helps with training, but is
  slightly more expensive in terms of computation and memory.

  Note that in dense implementation of this algorithm, mg, ms, and mom will
  update even if the grad is zero, but in this sparse implementation, mg, ms,
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  mean_grad = decay * mean_grad + (1-decay) * gradient

  Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)

  mg <- rho * mg_{t-1} + (1-rho) * grad
  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
  var <- var - mom

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    mg: A `Tensor` of type `resource`. Should be from a Variable().
    ms: A `Tensor` of type `resource`. Should be from a Variable().
    mom: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `lr`.
      Momentum Scale. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, mg, ms, and mom tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResourceApplyCenteredRMSProp', name, var, mg, ms, mom, lr, rho, momentum, epsilon, grad, 'use_locking', use_locking)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resource_apply_centered_rms_prop_eager_fallback(var, mg, ms, mom, lr, rho, momentum, epsilon, grad, use_locking=use_locking, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if use_locking is None:
        use_locking = False
    use_locking = _execute.make_bool(use_locking, 'use_locking')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResourceApplyCenteredRMSProp', var=var, mg=mg, ms=ms, mom=mom, lr=lr, rho=rho, momentum=momentum, epsilon=epsilon, grad=grad, use_locking=use_locking, name=name)
    return _op