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
def resource_sparse_apply_ftrl(var: _atypes.TensorFuzzingAnnotation[_atypes.Resource], accum: _atypes.TensorFuzzingAnnotation[_atypes.Resource], linear: _atypes.TensorFuzzingAnnotation[_atypes.Resource], grad: _atypes.TensorFuzzingAnnotation[TV_ResourceSparseApplyFtrl_T], indices: _atypes.TensorFuzzingAnnotation[TV_ResourceSparseApplyFtrl_Tindices], lr: _atypes.TensorFuzzingAnnotation[TV_ResourceSparseApplyFtrl_T], l1: _atypes.TensorFuzzingAnnotation[TV_ResourceSparseApplyFtrl_T], l2: _atypes.TensorFuzzingAnnotation[TV_ResourceSparseApplyFtrl_T], lr_power: _atypes.TensorFuzzingAnnotation[TV_ResourceSparseApplyFtrl_T], use_locking: bool=False, multiply_linear_by_lr: bool=False, name=None):
    """Update relevant entries in '*var' according to the Ftrl-proximal scheme.

  That is for rows we have grad for, we update var, accum and linear as follows:
  accum_new = accum + grad * grad
  linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
  quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
  var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
  accum = accum_new

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    linear: A `Tensor` of type `resource`. Should be from a Variable().
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    lr: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `grad`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `grad`.
      L2 regularization. Must be a scalar.
    lr_power: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    multiply_linear_by_lr: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResourceSparseApplyFtrl', name, var, accum, linear, grad, indices, lr, l1, l2, lr_power, 'use_locking', use_locking, 'multiply_linear_by_lr', multiply_linear_by_lr)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resource_sparse_apply_ftrl_eager_fallback(var, accum, linear, grad, indices, lr, l1, l2, lr_power, use_locking=use_locking, multiply_linear_by_lr=multiply_linear_by_lr, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if use_locking is None:
        use_locking = False
    use_locking = _execute.make_bool(use_locking, 'use_locking')
    if multiply_linear_by_lr is None:
        multiply_linear_by_lr = False
    multiply_linear_by_lr = _execute.make_bool(multiply_linear_by_lr, 'multiply_linear_by_lr')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResourceSparseApplyFtrl', var=var, accum=accum, linear=linear, grad=grad, indices=indices, lr=lr, l1=l1, l2=l2, lr_power=lr_power, use_locking=use_locking, multiply_linear_by_lr=multiply_linear_by_lr, name=name)
    return _op