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
def uniform_quantize(input: _atypes.TensorFuzzingAnnotation[TV_UniformQuantize_Tin], scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Tout: TV_UniformQuantize_Tout, quantization_min_val: int, quantization_max_val: int, quantization_axis: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[TV_UniformQuantize_Tout]:
    """Perform quantization on Tensor `input`.

  Given `input`, `scales` and `zero_points`, performs quantization using the formula:
  quantized_data = floor(input_data * (1.0f / scale) + 0.5f) + zero_point

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`.
      Must be a Tensor of Tin.
    scales: A `Tensor` of type `float32`.
      The float value(s) to use as scale(s) to quantize `input`.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
    zero_points: A `Tensor` of type `int32`.
      The int32 value(s) to use as zero_point(s) to quantize `input`.
      Same shape condition as scales.
    Tout: A `tf.DType` from: `tf.qint8, tf.qint32`.
      The type of output Tensor. A tf.DType from: tf.float32
    quantization_min_val: An `int`.
      The quantization min value to quantize `input`.
      The purpose of this attribute is typically (but not limited to) to indicate narrow range, where this is set to:
      `(Tin lowest) + 1` if narrow range, and `(Tin lowest)` otherwise.
      For example, if Tin is qint8, this is set to -127 if narrow range quantized or -128 if not.
    quantization_max_val: An `int`.
      The quantization max value to quantize `input`.
      The purpose of this attribute is typically (but not limited to) indicate narrow range, where this is set to:
      `(Tout max)` for both narrow range and not narrow range.
      For example, if Tin is qint8, this is set to 127.
    quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, input.dims()).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'UniformQuantize', name, input, scales, zero_points, 'Tout', Tout, 'quantization_axis', quantization_axis, 'quantization_min_val', quantization_min_val, 'quantization_max_val', quantization_max_val)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return uniform_quantize_eager_fallback(input, scales, zero_points, Tout=Tout, quantization_axis=quantization_axis, quantization_min_val=quantization_min_val, quantization_max_val=quantization_max_val, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    Tout = _execute.make_type(Tout, 'Tout')
    quantization_min_val = _execute.make_int(quantization_min_val, 'quantization_min_val')
    quantization_max_val = _execute.make_int(quantization_max_val, 'quantization_max_val')
    if quantization_axis is None:
        quantization_axis = -1
    quantization_axis = _execute.make_int(quantization_axis, 'quantization_axis')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('UniformQuantize', input=input, scales=scales, zero_points=zero_points, Tout=Tout, quantization_min_val=quantization_min_val, quantization_max_val=quantization_max_val, quantization_axis=quantization_axis, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tin', _op._get_attr_type('Tin'), 'Tout', _op._get_attr_type('Tout'), 'quantization_axis', _op._get_attr_int('quantization_axis'), 'quantization_min_val', _op._get_attr_int('quantization_min_val'), 'quantization_max_val', _op._get_attr_int('quantization_max_val'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('UniformQuantize', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result