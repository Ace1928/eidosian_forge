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
def sparse_apply_proximal_gradient_descent(var: _atypes.TensorFuzzingAnnotation[TV_SparseApplyProximalGradientDescent_T], alpha: _atypes.TensorFuzzingAnnotation[TV_SparseApplyProximalGradientDescent_T], l1: _atypes.TensorFuzzingAnnotation[TV_SparseApplyProximalGradientDescent_T], l2: _atypes.TensorFuzzingAnnotation[TV_SparseApplyProximalGradientDescent_T], grad: _atypes.TensorFuzzingAnnotation[TV_SparseApplyProximalGradientDescent_T], indices: _atypes.TensorFuzzingAnnotation[TV_SparseApplyProximalGradientDescent_Tindices], use_locking: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_SparseApplyProximalGradientDescent_T]:
    """Sparse update '*var' as FOBOS algorithm with fixed learning rate.

  That is for rows we have grad for, we update var as follows:
  $$prox_v = var - alpha * grad$$
  $$var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}$$

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    alpha: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("sparse_apply_proximal_gradient_descent op does not support eager execution. Arg 'out' is a ref.")
    if use_locking is None:
        use_locking = False
    use_locking = _execute.make_bool(use_locking, 'use_locking')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseApplyProximalGradientDescent', var=var, alpha=alpha, l1=l1, l2=l2, grad=grad, indices=indices, use_locking=use_locking, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'), 'use_locking', _op._get_attr_bool('use_locking'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseApplyProximalGradientDescent', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result