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
def quantized_batch_norm_with_global_normalization_eager_fallback(t: _atypes.TensorFuzzingAnnotation[TV_QuantizedBatchNormWithGlobalNormalization_Tinput], t_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], t_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], m: _atypes.TensorFuzzingAnnotation[TV_QuantizedBatchNormWithGlobalNormalization_Tinput], m_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], m_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], v: _atypes.TensorFuzzingAnnotation[TV_QuantizedBatchNormWithGlobalNormalization_Tinput], v_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], v_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], beta: _atypes.TensorFuzzingAnnotation[TV_QuantizedBatchNormWithGlobalNormalization_Tinput], beta_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], beta_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], gamma: _atypes.TensorFuzzingAnnotation[TV_QuantizedBatchNormWithGlobalNormalization_Tinput], gamma_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], gamma_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], out_type: TV_QuantizedBatchNormWithGlobalNormalization_out_type, variance_epsilon: float, scale_after_normalization: bool, name, ctx):
    out_type = _execute.make_type(out_type, 'out_type')
    variance_epsilon = _execute.make_float(variance_epsilon, 'variance_epsilon')
    scale_after_normalization = _execute.make_bool(scale_after_normalization, 'scale_after_normalization')
    _attr_Tinput, _inputs_Tinput = _execute.args_to_matching_eager([t, m, v, beta, gamma], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    t, m, v, beta, gamma = _inputs_Tinput
    t_min = _ops.convert_to_tensor(t_min, _dtypes.float32)
    t_max = _ops.convert_to_tensor(t_max, _dtypes.float32)
    m_min = _ops.convert_to_tensor(m_min, _dtypes.float32)
    m_max = _ops.convert_to_tensor(m_max, _dtypes.float32)
    v_min = _ops.convert_to_tensor(v_min, _dtypes.float32)
    v_max = _ops.convert_to_tensor(v_max, _dtypes.float32)
    beta_min = _ops.convert_to_tensor(beta_min, _dtypes.float32)
    beta_max = _ops.convert_to_tensor(beta_max, _dtypes.float32)
    gamma_min = _ops.convert_to_tensor(gamma_min, _dtypes.float32)
    gamma_max = _ops.convert_to_tensor(gamma_max, _dtypes.float32)
    _inputs_flat = [t, t_min, t_max, m, m_min, m_max, v, v_min, v_max, beta, beta_min, beta_max, gamma, gamma_min, gamma_max]
    _attrs = ('Tinput', _attr_Tinput, 'out_type', out_type, 'variance_epsilon', variance_epsilon, 'scale_after_normalization', scale_after_normalization)
    _result = _execute.execute(b'QuantizedBatchNormWithGlobalNormalization', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizedBatchNormWithGlobalNormalization', _inputs_flat, _attrs, _result)
    _result = _QuantizedBatchNormWithGlobalNormalizationOutput._make(_result)
    return _result