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
def uniform_quantized_convolution_eager_fallback(lhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedConvolution_Tin], rhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedConvolution_Tin], lhs_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], lhs_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], rhs_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], rhs_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], output_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], output_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Tout: TV_UniformQuantizedConvolution_Tout, padding: str, lhs_quantization_min_val: int, lhs_quantization_max_val: int, rhs_quantization_min_val: int, rhs_quantization_max_val: int, output_quantization_min_val: int, output_quantization_max_val: int, window_strides, explicit_padding, lhs_dilation, rhs_dilation, batch_group_count: int, feature_group_count: int, dimension_numbers: str, lhs_quantization_axis: int, rhs_quantization_axis: int, output_quantization_axis: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedConvolution_Tout]:
    Tout = _execute.make_type(Tout, 'Tout')
    padding = _execute.make_str(padding, 'padding')
    lhs_quantization_min_val = _execute.make_int(lhs_quantization_min_val, 'lhs_quantization_min_val')
    lhs_quantization_max_val = _execute.make_int(lhs_quantization_max_val, 'lhs_quantization_max_val')
    rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, 'rhs_quantization_min_val')
    rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, 'rhs_quantization_max_val')
    output_quantization_min_val = _execute.make_int(output_quantization_min_val, 'output_quantization_min_val')
    output_quantization_max_val = _execute.make_int(output_quantization_max_val, 'output_quantization_max_val')
    if window_strides is None:
        window_strides = []
    if not isinstance(window_strides, (list, tuple)):
        raise TypeError("Expected list for 'window_strides' argument to 'uniform_quantized_convolution' Op, not %r." % window_strides)
    window_strides = [_execute.make_int(_i, 'window_strides') for _i in window_strides]
    if explicit_padding is None:
        explicit_padding = []
    if not isinstance(explicit_padding, (list, tuple)):
        raise TypeError("Expected list for 'explicit_padding' argument to 'uniform_quantized_convolution' Op, not %r." % explicit_padding)
    explicit_padding = [_execute.make_int(_i, 'explicit_padding') for _i in explicit_padding]
    if lhs_dilation is None:
        lhs_dilation = []
    if not isinstance(lhs_dilation, (list, tuple)):
        raise TypeError("Expected list for 'lhs_dilation' argument to 'uniform_quantized_convolution' Op, not %r." % lhs_dilation)
    lhs_dilation = [_execute.make_int(_i, 'lhs_dilation') for _i in lhs_dilation]
    if rhs_dilation is None:
        rhs_dilation = []
    if not isinstance(rhs_dilation, (list, tuple)):
        raise TypeError("Expected list for 'rhs_dilation' argument to 'uniform_quantized_convolution' Op, not %r." % rhs_dilation)
    rhs_dilation = [_execute.make_int(_i, 'rhs_dilation') for _i in rhs_dilation]
    if batch_group_count is None:
        batch_group_count = 1
    batch_group_count = _execute.make_int(batch_group_count, 'batch_group_count')
    if feature_group_count is None:
        feature_group_count = 1
    feature_group_count = _execute.make_int(feature_group_count, 'feature_group_count')
    if dimension_numbers is None:
        dimension_numbers = ''
    dimension_numbers = _execute.make_str(dimension_numbers, 'dimension_numbers')
    if lhs_quantization_axis is None:
        lhs_quantization_axis = -1
    lhs_quantization_axis = _execute.make_int(lhs_quantization_axis, 'lhs_quantization_axis')
    if rhs_quantization_axis is None:
        rhs_quantization_axis = -1
    rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, 'rhs_quantization_axis')
    if output_quantization_axis is None:
        output_quantization_axis = -1
    output_quantization_axis = _execute.make_int(output_quantization_axis, 'output_quantization_axis')
    _attr_Tin, _inputs_Tin = _execute.args_to_matching_eager([lhs, rhs], ctx, [_dtypes.qint8])
    lhs, rhs = _inputs_Tin
    lhs_scales = _ops.convert_to_tensor(lhs_scales, _dtypes.float32)
    lhs_zero_points = _ops.convert_to_tensor(lhs_zero_points, _dtypes.int32)
    rhs_scales = _ops.convert_to_tensor(rhs_scales, _dtypes.float32)
    rhs_zero_points = _ops.convert_to_tensor(rhs_zero_points, _dtypes.int32)
    output_scales = _ops.convert_to_tensor(output_scales, _dtypes.float32)
    output_zero_points = _ops.convert_to_tensor(output_zero_points, _dtypes.int32)
    _inputs_flat = [lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points, output_scales, output_zero_points]
    _attrs = ('Tin', _attr_Tin, 'Tout', Tout, 'window_strides', window_strides, 'padding', padding, 'explicit_padding', explicit_padding, 'lhs_dilation', lhs_dilation, 'rhs_dilation', rhs_dilation, 'batch_group_count', batch_group_count, 'feature_group_count', feature_group_count, 'dimension_numbers', dimension_numbers, 'lhs_quantization_axis', lhs_quantization_axis, 'lhs_quantization_min_val', lhs_quantization_min_val, 'lhs_quantization_max_val', lhs_quantization_max_val, 'rhs_quantization_axis', rhs_quantization_axis, 'rhs_quantization_min_val', rhs_quantization_min_val, 'rhs_quantization_max_val', rhs_quantization_max_val, 'output_quantization_axis', output_quantization_axis, 'output_quantization_min_val', output_quantization_min_val, 'output_quantization_max_val', output_quantization_max_val)
    _result = _execute.execute(b'UniformQuantizedConvolution', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('UniformQuantizedConvolution', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result