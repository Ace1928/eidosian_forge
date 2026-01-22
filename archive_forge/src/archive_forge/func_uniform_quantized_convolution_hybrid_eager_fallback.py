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
def uniform_quantized_convolution_hybrid_eager_fallback(lhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedConvolutionHybrid_Tlhs], rhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedConvolutionHybrid_Trhs], rhs_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], rhs_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Tout: TV_UniformQuantizedConvolutionHybrid_Tout, padding: str, rhs_quantization_min_val: int, rhs_quantization_max_val: int, window_strides, explicit_padding, lhs_dilation, rhs_dilation, batch_group_count: int, feature_group_count: int, dimension_numbers: str, rhs_quantization_axis: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedConvolutionHybrid_Tout]:
    Tout = _execute.make_type(Tout, 'Tout')
    padding = _execute.make_str(padding, 'padding')
    rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, 'rhs_quantization_min_val')
    rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, 'rhs_quantization_max_val')
    if window_strides is None:
        window_strides = []
    if not isinstance(window_strides, (list, tuple)):
        raise TypeError("Expected list for 'window_strides' argument to 'uniform_quantized_convolution_hybrid' Op, not %r." % window_strides)
    window_strides = [_execute.make_int(_i, 'window_strides') for _i in window_strides]
    if explicit_padding is None:
        explicit_padding = []
    if not isinstance(explicit_padding, (list, tuple)):
        raise TypeError("Expected list for 'explicit_padding' argument to 'uniform_quantized_convolution_hybrid' Op, not %r." % explicit_padding)
    explicit_padding = [_execute.make_int(_i, 'explicit_padding') for _i in explicit_padding]
    if lhs_dilation is None:
        lhs_dilation = []
    if not isinstance(lhs_dilation, (list, tuple)):
        raise TypeError("Expected list for 'lhs_dilation' argument to 'uniform_quantized_convolution_hybrid' Op, not %r." % lhs_dilation)
    lhs_dilation = [_execute.make_int(_i, 'lhs_dilation') for _i in lhs_dilation]
    if rhs_dilation is None:
        rhs_dilation = []
    if not isinstance(rhs_dilation, (list, tuple)):
        raise TypeError("Expected list for 'rhs_dilation' argument to 'uniform_quantized_convolution_hybrid' Op, not %r." % rhs_dilation)
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
    if rhs_quantization_axis is None:
        rhs_quantization_axis = -1
    rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, 'rhs_quantization_axis')
    _attr_Tlhs, (lhs,) = _execute.args_to_matching_eager([lhs], ctx, [_dtypes.float32])
    _attr_Trhs, (rhs,) = _execute.args_to_matching_eager([rhs], ctx, [_dtypes.qint8])
    rhs_scales = _ops.convert_to_tensor(rhs_scales, _dtypes.float32)
    rhs_zero_points = _ops.convert_to_tensor(rhs_zero_points, _dtypes.int32)
    _inputs_flat = [lhs, rhs, rhs_scales, rhs_zero_points]
    _attrs = ('Tlhs', _attr_Tlhs, 'Trhs', _attr_Trhs, 'Tout', Tout, 'window_strides', window_strides, 'padding', padding, 'explicit_padding', explicit_padding, 'lhs_dilation', lhs_dilation, 'rhs_dilation', rhs_dilation, 'batch_group_count', batch_group_count, 'feature_group_count', feature_group_count, 'dimension_numbers', dimension_numbers, 'rhs_quantization_axis', rhs_quantization_axis, 'rhs_quantization_min_val', rhs_quantization_min_val, 'rhs_quantization_max_val', rhs_quantization_max_val)
    _result = _execute.execute(b'UniformQuantizedConvolutionHybrid', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('UniformQuantizedConvolutionHybrid', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result