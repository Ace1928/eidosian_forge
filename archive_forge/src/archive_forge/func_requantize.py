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
def requantize(input: _atypes.TensorFuzzingAnnotation[TV_Requantize_Tinput], input_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], input_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], requested_output_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], requested_output_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], out_type: TV_Requantize_out_type, name=None):
    """Converts the quantized `input` tensor into a lower-precision `output`.

  Converts the quantized `input` tensor into a lower-precision `output`, using the
  output range specified with `requested_output_min` and `requested_output_max`.

  `[input_min, input_max]` are scalar floats that specify the range for the float
  interpretation of the `input` data. For example, if `input_min` is -1.0f and
  `input_max` is 1.0f, and we are dealing with `quint16` quantized data, then a 0
  value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    requested_output_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized output value represents.
    requested_output_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized output value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
      The type of the output. Should be a lower bit depth than Tinput.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `out_type`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Requantize', name, input, input_min, input_max, requested_output_min, requested_output_max, 'out_type', out_type)
            _result = _RequantizeOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return requantize_eager_fallback(input, input_min, input_max, requested_output_min, requested_output_max, out_type=out_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    out_type = _execute.make_type(out_type, 'out_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Requantize', input=input, input_min=input_min, input_max=input_max, requested_output_min=requested_output_min, requested_output_max=requested_output_max, out_type=out_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tinput', _op._get_attr_type('Tinput'), 'out_type', _op._get_attr_type('out_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Requantize', _inputs_flat, _attrs, _result)
    _result = _RequantizeOutput._make(_result)
    return _result