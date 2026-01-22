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
def quantized_bias_add(input: _atypes.TensorFuzzingAnnotation[TV_QuantizedBiasAdd_T1], bias: _atypes.TensorFuzzingAnnotation[TV_QuantizedBiasAdd_T2], min_input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_bias: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_bias: _atypes.TensorFuzzingAnnotation[_atypes.Float32], out_type: TV_QuantizedBiasAdd_out_type, name=None):
    """Adds Tensor 'bias' to Tensor 'input' for Quantized types.

  Broadcasts the values of bias on dimensions 0..N-2 of 'input'.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A 1D bias Tensor with size matching the last dimension of 'input'.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    min_bias: A `Tensor` of type `float32`.
      The float value that the lowest quantized bias value represents.
    max_bias: A `Tensor` of type `float32`.
      The float value that the highest quantized bias value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_out, max_out).

    output: A `Tensor` of type `out_type`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizedBiasAdd', name, input, bias, min_input, max_input, min_bias, max_bias, 'out_type', out_type)
            _result = _QuantizedBiasAddOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantized_bias_add_eager_fallback(input, bias, min_input, max_input, min_bias, max_bias, out_type=out_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    out_type = _execute.make_type(out_type, 'out_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizedBiasAdd', input=input, bias=bias, min_input=min_input, max_input=max_input, min_bias=min_bias, max_bias=max_bias, out_type=out_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T1', _op._get_attr_type('T1'), 'T2', _op._get_attr_type('T2'), 'out_type', _op._get_attr_type('out_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizedBiasAdd', _inputs_flat, _attrs, _result)
    _result = _QuantizedBiasAddOutput._make(_result)
    return _result