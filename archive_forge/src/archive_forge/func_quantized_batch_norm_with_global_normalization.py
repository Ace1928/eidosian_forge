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
def quantized_batch_norm_with_global_normalization(t: _atypes.TensorFuzzingAnnotation[TV_QuantizedBatchNormWithGlobalNormalization_Tinput], t_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], t_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], m: _atypes.TensorFuzzingAnnotation[TV_QuantizedBatchNormWithGlobalNormalization_Tinput], m_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], m_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], v: _atypes.TensorFuzzingAnnotation[TV_QuantizedBatchNormWithGlobalNormalization_Tinput], v_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], v_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], beta: _atypes.TensorFuzzingAnnotation[TV_QuantizedBatchNormWithGlobalNormalization_Tinput], beta_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], beta_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], gamma: _atypes.TensorFuzzingAnnotation[TV_QuantizedBatchNormWithGlobalNormalization_Tinput], gamma_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], gamma_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], out_type: TV_QuantizedBatchNormWithGlobalNormalization_out_type, variance_epsilon: float, scale_after_normalization: bool, name=None):
    """Quantized Batch normalization.

  This op is deprecated and will be removed in the future. Prefer
  `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A 4D input Tensor.
    t_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized input.
    t_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized input.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    m_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized mean.
    m_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized mean.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    v_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized variance.
    v_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized variance.
    beta: A `Tensor`. Must have the same type as `t`.
      A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    beta_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized offset.
    beta_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized offset.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    gamma_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized gamma.
    gamma_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized gamma.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result, result_min, result_max).

    result: A `Tensor` of type `out_type`.
    result_min: A `Tensor` of type `float32`.
    result_max: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizedBatchNormWithGlobalNormalization', name, t, t_min, t_max, m, m_min, m_max, v, v_min, v_max, beta, beta_min, beta_max, gamma, gamma_min, gamma_max, 'out_type', out_type, 'variance_epsilon', variance_epsilon, 'scale_after_normalization', scale_after_normalization)
            _result = _QuantizedBatchNormWithGlobalNormalizationOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantized_batch_norm_with_global_normalization_eager_fallback(t, t_min, t_max, m, m_min, m_max, v, v_min, v_max, beta, beta_min, beta_max, gamma, gamma_min, gamma_max, out_type=out_type, variance_epsilon=variance_epsilon, scale_after_normalization=scale_after_normalization, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    out_type = _execute.make_type(out_type, 'out_type')
    variance_epsilon = _execute.make_float(variance_epsilon, 'variance_epsilon')
    scale_after_normalization = _execute.make_bool(scale_after_normalization, 'scale_after_normalization')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizedBatchNormWithGlobalNormalization', t=t, t_min=t_min, t_max=t_max, m=m, m_min=m_min, m_max=m_max, v=v, v_min=v_min, v_max=v_max, beta=beta, beta_min=beta_min, beta_max=beta_max, gamma=gamma, gamma_min=gamma_min, gamma_max=gamma_max, out_type=out_type, variance_epsilon=variance_epsilon, scale_after_normalization=scale_after_normalization, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tinput', _op._get_attr_type('Tinput'), 'out_type', _op._get_attr_type('out_type'), 'variance_epsilon', _op.get_attr('variance_epsilon'), 'scale_after_normalization', _op._get_attr_bool('scale_after_normalization'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizedBatchNormWithGlobalNormalization', _inputs_flat, _attrs, _result)
    _result = _QuantizedBatchNormWithGlobalNormalizationOutput._make(_result)
    return _result