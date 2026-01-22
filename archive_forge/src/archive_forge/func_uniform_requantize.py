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
def uniform_requantize(input: _atypes.TensorFuzzingAnnotation[TV_UniformRequantize_Tin], input_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], input_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], output_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], output_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Tout: TV_UniformRequantize_Tout, input_quantization_min_val: int, input_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, input_quantization_axis: int=-1, output_quantization_axis: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[TV_UniformRequantize_Tout]:
    """Given quantized tensor `input`, requantize it with new quantization parameters.

  Given quantized tensor `input`, which was quantized using {input_scales, input_zero_points, input_quantization_axis, input_quantization_min_val, input_quantization_max_val},
  requantize it to a tensor, which is quantized using {output_scales, output_zero_points, output_quantization_axis, output_quantization_min_val, output_quantization_max_val}.
  The requantization is done by using the formula:
  output_quantized_data = clip(
    (input_quantized_data - input_zero_point) * (input_scale / output_scale) + output_zero_point,
    output_quantization_min_val,
    output_quantization_max_val)

  Per-tensor and per-axis quantization supported cases are followings:
  * per-tensor -> per-tensor
  * per-tensor -> per-axis
  * per-axis -> per-axis where input_quantization_axis equals output_quantization_axis.
  i.e. At least one among input_quantization_axis and output_quantization_axis must be -1, or two must be equal.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `qint32`.
      Must be a Tensor of Tin.
    input_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale(s) when quantizing original data that `input` represents.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
    input_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point(s) when quantizing original data that `input` represents.
      Same shape condition as scales.
    output_scales: A `Tensor` of type `float32`.
      The float value(s) to use as new scale(s) to quantize original data that `input` represents.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
    output_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) to use as new zero_point(s) to quantize original data that `input` represents.
      Same shape condition as scales.
    Tout: A `tf.DType` from: `tf.qint8, tf.qint32`.
      The type of output Tensor. A tf.DType from: tf.qint8, tf.qint32
    input_quantization_min_val: An `int`.
      The quantization min value that was used when quantizing original data that `input` represents.
      The purpose of this attribute is typically (but not limited to) to indicate narrow range, where this is set to:
      `(Tin lowest) + 1` if narrow range, and `(Tin lowest)` otherwise.
      For example, if Tin is qint8, this is set to -127 if narrow range quantized or -128 if not.
    input_quantization_max_val: An `int`.
      The quantization max value that was used when quantizing original data that `input` represents.
      The purpose of this attribute is typically (but not limited to) indicate narrow range, where this is set to:
      `(Tout max)` for both narrow range and not narrow range.
      For example, if Tin is qint8, this is set to 127.
    output_quantization_min_val: An `int`.
      The new quantization min value to quantize original data that `input` represents.
    output_quantization_max_val: An `int`.
      The new quantization max value to quantize original data that `input` represents.
    input_quantization_axis: An optional `int`. Defaults to `-1`.
      The quantization axis that was used when quantizing original data that `input` represents.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, input.dims()).
    output_quantization_axis: An optional `int`. Defaults to `-1`.
      The new quantization axis to use to quantize original data that `input` represents.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'UniformRequantize', name, input, input_scales, input_zero_points, output_scales, output_zero_points, 'Tout', Tout, 'input_quantization_axis', input_quantization_axis, 'input_quantization_min_val', input_quantization_min_val, 'input_quantization_max_val', input_quantization_max_val, 'output_quantization_axis', output_quantization_axis, 'output_quantization_min_val', output_quantization_min_val, 'output_quantization_max_val', output_quantization_max_val)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return uniform_requantize_eager_fallback(input, input_scales, input_zero_points, output_scales, output_zero_points, Tout=Tout, input_quantization_axis=input_quantization_axis, input_quantization_min_val=input_quantization_min_val, input_quantization_max_val=input_quantization_max_val, output_quantization_axis=output_quantization_axis, output_quantization_min_val=output_quantization_min_val, output_quantization_max_val=output_quantization_max_val, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    Tout = _execute.make_type(Tout, 'Tout')
    input_quantization_min_val = _execute.make_int(input_quantization_min_val, 'input_quantization_min_val')
    input_quantization_max_val = _execute.make_int(input_quantization_max_val, 'input_quantization_max_val')
    output_quantization_min_val = _execute.make_int(output_quantization_min_val, 'output_quantization_min_val')
    output_quantization_max_val = _execute.make_int(output_quantization_max_val, 'output_quantization_max_val')
    if input_quantization_axis is None:
        input_quantization_axis = -1
    input_quantization_axis = _execute.make_int(input_quantization_axis, 'input_quantization_axis')
    if output_quantization_axis is None:
        output_quantization_axis = -1
    output_quantization_axis = _execute.make_int(output_quantization_axis, 'output_quantization_axis')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('UniformRequantize', input=input, input_scales=input_scales, input_zero_points=input_zero_points, output_scales=output_scales, output_zero_points=output_zero_points, Tout=Tout, input_quantization_min_val=input_quantization_min_val, input_quantization_max_val=input_quantization_max_val, output_quantization_min_val=output_quantization_min_val, output_quantization_max_val=output_quantization_max_val, input_quantization_axis=input_quantization_axis, output_quantization_axis=output_quantization_axis, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tin', _op._get_attr_type('Tin'), 'Tout', _op._get_attr_type('Tout'), 'input_quantization_axis', _op._get_attr_int('input_quantization_axis'), 'input_quantization_min_val', _op._get_attr_int('input_quantization_min_val'), 'input_quantization_max_val', _op._get_attr_int('input_quantization_max_val'), 'output_quantization_axis', _op._get_attr_int('output_quantization_axis'), 'output_quantization_min_val', _op._get_attr_int('output_quantization_min_val'), 'output_quantization_max_val', _op._get_attr_int('output_quantization_max_val'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('UniformRequantize', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result