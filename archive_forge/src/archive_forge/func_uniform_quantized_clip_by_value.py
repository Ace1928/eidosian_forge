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
def uniform_quantized_clip_by_value(operand: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedClipByValue_T], min: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedClipByValue_T], max: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedClipByValue_T], scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], quantization_min_val: int, quantization_max_val: int, quantization_axis: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedClipByValue_T]:
    """Perform clip by value on the quantized Tensor `operand`.

  Given quantized `operand` which was quantized using `scales` and `zero_points`, performs clip by value using `min` and `max` values.
  If quantization_axis is -1 (per-tensor quantized), the entire operand is clipped using scalar min, max.
  Otherwise (per-channel quantized), the clipping is also done per-channel.

  Args:
    operand: A `Tensor`. Must be one of the following types: `qint32`.
      Must be a Tensor of T.
    min: A `Tensor`. Must have the same type as `operand`.
      The min value(s) to clip operand. Must be a Tensor of T.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (operand.dim_size(quantization_axis),) (per-axis quantization).
    max: A `Tensor`. Must have the same type as `operand`.
      The min value(s) to clip operand. Must be a Tensor of T.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (operand.dim_size(quantization_axis),) (per-axis quantization).
    scales: A `Tensor` of type `float32`.
      The float value(s) used as scale(s) when quantizing `operand`, `min` and `max`.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (operand.dim_size(quantization_axis),) (per-axis quantization).
    zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point(s) when quantizing `operand`, `min` and `max`.
      Same shape condition as scales.
    quantization_min_val: An `int`.
      The quantization min value that was used when operand was quantized.
    quantization_max_val: An `int`.
      The quantization max value that was used when operand was quantized.
    quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, operand.dims()).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `operand`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'UniformQuantizedClipByValue', name, operand, min, max, scales, zero_points, 'quantization_axis', quantization_axis, 'quantization_min_val', quantization_min_val, 'quantization_max_val', quantization_max_val)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return uniform_quantized_clip_by_value_eager_fallback(operand, min, max, scales, zero_points, quantization_axis=quantization_axis, quantization_min_val=quantization_min_val, quantization_max_val=quantization_max_val, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    quantization_min_val = _execute.make_int(quantization_min_val, 'quantization_min_val')
    quantization_max_val = _execute.make_int(quantization_max_val, 'quantization_max_val')
    if quantization_axis is None:
        quantization_axis = -1
    quantization_axis = _execute.make_int(quantization_axis, 'quantization_axis')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('UniformQuantizedClipByValue', operand=operand, min=min, max=max, scales=scales, zero_points=zero_points, quantization_min_val=quantization_min_val, quantization_max_val=quantization_max_val, quantization_axis=quantization_axis, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'quantization_axis', _op._get_attr_int('quantization_axis'), 'quantization_min_val', _op._get_attr_int('quantization_min_val'), 'quantization_max_val', _op._get_attr_int('quantization_max_val'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('UniformQuantizedClipByValue', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result