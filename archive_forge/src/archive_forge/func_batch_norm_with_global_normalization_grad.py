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
def batch_norm_with_global_normalization_grad(t: _atypes.TensorFuzzingAnnotation[TV_BatchNormWithGlobalNormalizationGrad_T], m: _atypes.TensorFuzzingAnnotation[TV_BatchNormWithGlobalNormalizationGrad_T], v: _atypes.TensorFuzzingAnnotation[TV_BatchNormWithGlobalNormalizationGrad_T], gamma: _atypes.TensorFuzzingAnnotation[TV_BatchNormWithGlobalNormalizationGrad_T], backprop: _atypes.TensorFuzzingAnnotation[TV_BatchNormWithGlobalNormalizationGrad_T], variance_epsilon: float, scale_after_normalization: bool, name=None):
    """Gradients for batch normalization.

  This op is deprecated. See `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A 4D input Tensor.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this Tensor will be multiplied
      with the normalized Tensor.
    backprop: A `Tensor`. Must have the same type as `t`. 4D backprop Tensor.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (dx, dm, dv, db, dg).

    dx: A `Tensor`. Has the same type as `t`.
    dm: A `Tensor`. Has the same type as `t`.
    dv: A `Tensor`. Has the same type as `t`.
    db: A `Tensor`. Has the same type as `t`.
    dg: A `Tensor`. Has the same type as `t`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BatchNormWithGlobalNormalizationGrad', name, t, m, v, gamma, backprop, 'variance_epsilon', variance_epsilon, 'scale_after_normalization', scale_after_normalization)
            _result = _BatchNormWithGlobalNormalizationGradOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return batch_norm_with_global_normalization_grad_eager_fallback(t, m, v, gamma, backprop, variance_epsilon=variance_epsilon, scale_after_normalization=scale_after_normalization, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    variance_epsilon = _execute.make_float(variance_epsilon, 'variance_epsilon')
    scale_after_normalization = _execute.make_bool(scale_after_normalization, 'scale_after_normalization')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BatchNormWithGlobalNormalizationGrad', t=t, m=m, v=v, gamma=gamma, backprop=backprop, variance_epsilon=variance_epsilon, scale_after_normalization=scale_after_normalization, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'variance_epsilon', _op.get_attr('variance_epsilon'), 'scale_after_normalization', _op._get_attr_bool('scale_after_normalization'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BatchNormWithGlobalNormalizationGrad', _inputs_flat, _attrs, _result)
    _result = _BatchNormWithGlobalNormalizationGradOutput._make(_result)
    return _result