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
def quantized_relu_x(features: _atypes.TensorFuzzingAnnotation[TV_QuantizedReluX_Tinput], max_value: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_features: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_features: _atypes.TensorFuzzingAnnotation[_atypes.Float32], out_type: TV_QuantizedReluX_out_type=_dtypes.quint8, name=None):
    """Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    max_value: A `Tensor` of type `float32`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`.
    min_activations: A `Tensor` of type `float32`.
    max_activations: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizedReluX', name, features, max_value, min_features, max_features, 'out_type', out_type)
            _result = _QuantizedReluXOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantized_relu_x_eager_fallback(features, max_value, min_features, max_features, out_type=out_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if out_type is None:
        out_type = _dtypes.quint8
    out_type = _execute.make_type(out_type, 'out_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizedReluX', features=features, max_value=max_value, min_features=min_features, max_features=max_features, out_type=out_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tinput', _op._get_attr_type('Tinput'), 'out_type', _op._get_attr_type('out_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizedReluX', _inputs_flat, _attrs, _result)
    _result = _QuantizedReluXOutput._make(_result)
    return _result