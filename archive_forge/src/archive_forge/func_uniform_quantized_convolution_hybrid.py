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
def uniform_quantized_convolution_hybrid(lhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedConvolutionHybrid_Tlhs], rhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedConvolutionHybrid_Trhs], rhs_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], rhs_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Tout: TV_UniformQuantizedConvolutionHybrid_Tout, padding: str, rhs_quantization_min_val: int, rhs_quantization_max_val: int, window_strides=[], explicit_padding=[], lhs_dilation=[], rhs_dilation=[], batch_group_count: int=1, feature_group_count: int=1, dimension_numbers: str='', rhs_quantization_axis: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedConvolutionHybrid_Tout]:
    """Perform hybrid quantized convolution of float Tensor `lhs` and quantized Tensor `rhs`.

  Given float `lhs` and quantized `rhs`, internally performs quantization on `lhs`,
  and then performs quantized convolution on quantized `lhs` and `rhs`.

  The internal quantization on `lhs` is a quantization to `Trhs`, dynamic range,
  per-batch (per-axis along axis `dimension_numbers.input_batch_dimension`), asymmetric,
  and not narrow range (the range is [Trhs_MIN, Trhs_MAX]).

  `lhs` and `rhs` must be Tensors of same rank, and meet following shape conditions.
  - lhs_feature % feature_group_count == 0
  - lhs_feature % rhs_input_feature == 0
  - lhs_feature / feature_group_count == rhs_input_feature
  - rhs_output_feature % feature_group_count == 0
  - lhs_batch % batch_group_count == 0
  - rhs_output_feature % batch_group_count == 0

  `rhs` must be quantized Tensor, where its data value is quantized using the formula:
  quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val).

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`.
      Must be a non-quantized Tensor of `Tlhs`, rank >= 3.
    rhs: A `Tensor`. Must be one of the following types: `qint8`.
      Must be a quantized Tensor of `Trhs`, same rank as `lhs`.
    rhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale factors when quantizing the original data that `rhs` represents.
      Must be a scalar Tensor for per-tensor quantization,
      or 1D Tensor of size `rhs.dim_size(kernel_output_feature_dimension)`, for per-channel quantization.
    rhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that `rhs` represents.
      Same shape condition as `rhs_scales`.
    Tout: A `tf.DType` from: `tf.float32`. The type of output Tensor.
    padding: A `string`.
      string from: `"SAME"`, `"VALID"`, or `"EXPLICIT"`, indicating the type of padding algorithm to use.
    rhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in `rhs`.
      For example, if `Trhs` is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    rhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in `rhs`.
      For example, if `Trhs` is qint8, this must be set to 127.
    window_strides: An optional list of `ints`. Defaults to `[]`.
      The stride of the sliding window for each spatial dimension of `lhs`.
      Must be an empty list (default) or a list of size (number of spatial dimensions).
      If an empty list is provided, the stride for each spatial dimension is set to 1.
    explicit_padding: An optional list of `ints`. Defaults to `[]`.
      If `padding` Attr is `"EXPLICIT"`, must be set as a list indicating
      the explicit paddings at the start and end of each lhs spatial dimension.
      Otherwise, this Attr is must be empty.

      (If used,) Must be a list of size 2 * (number of lhs spatial dimensions),
      where (explicit_padding[2 * i], explicit_padding[2 * i + 1]) indicates
      spatial_dimensions[i] (start_padding, end_padding).
    lhs_dilation: An optional list of `ints`. Defaults to `[]`.
      The dilation factor to apply in each spatial dimension of `lhs`.
      Must be an empty list (default) or a list of size (number of lhs spatial dimensions).
      If empty list, the dilation for each lhs spatial dimension is set to 1.
    rhs_dilation: An optional list of `ints`. Defaults to `[]`.
      The dilation factor to apply in each spatial dimension of `rhs`.
      Must be an empty list (default) or a list of size (number of rhs spatial dimensions).
      If empty list, the dilation for each rhs spatial dimension is set to 1.
    batch_group_count: An optional `int`. Defaults to `1`.
      The number of batch groups. Used for grouped filters.
      Must be a divisor of output_feature.
    feature_group_count: An optional `int`. Defaults to `1`.
      The number of feature groups. Used for grouped convolutions.
      Must be a divisor of both lhs_feature and output_feature.
    dimension_numbers: An optional `string`. Defaults to `""`.
      Structure of dimension information for the convolution op.
      Must be an empty string (default) or a serialized string of tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr proto.
      If empty string, the default is `("NCHW", "OIHW", "NCHW")` (for a 2D convolution).
    rhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For the `rhs`, only per-tensor quantization
      or per-channel quantization along kernel_output_feature_dimension is supported.
      Thus, this attribute must be set to -1 or `dimension_numbers.kernel_output_feature_dimension`.
      Other values will raise error at OpKernel construction.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'UniformQuantizedConvolutionHybrid', name, lhs, rhs, rhs_scales, rhs_zero_points, 'Tout', Tout, 'window_strides', window_strides, 'padding', padding, 'explicit_padding', explicit_padding, 'lhs_dilation', lhs_dilation, 'rhs_dilation', rhs_dilation, 'batch_group_count', batch_group_count, 'feature_group_count', feature_group_count, 'dimension_numbers', dimension_numbers, 'rhs_quantization_axis', rhs_quantization_axis, 'rhs_quantization_min_val', rhs_quantization_min_val, 'rhs_quantization_max_val', rhs_quantization_max_val)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return uniform_quantized_convolution_hybrid_eager_fallback(lhs, rhs, rhs_scales, rhs_zero_points, Tout=Tout, window_strides=window_strides, padding=padding, explicit_padding=explicit_padding, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation, batch_group_count=batch_group_count, feature_group_count=feature_group_count, dimension_numbers=dimension_numbers, rhs_quantization_axis=rhs_quantization_axis, rhs_quantization_min_val=rhs_quantization_min_val, rhs_quantization_max_val=rhs_quantization_max_val, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
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
    _, _, _op, _outputs = _op_def_library._apply_op_helper('UniformQuantizedConvolutionHybrid', lhs=lhs, rhs=rhs, rhs_scales=rhs_scales, rhs_zero_points=rhs_zero_points, Tout=Tout, padding=padding, rhs_quantization_min_val=rhs_quantization_min_val, rhs_quantization_max_val=rhs_quantization_max_val, window_strides=window_strides, explicit_padding=explicit_padding, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation, batch_group_count=batch_group_count, feature_group_count=feature_group_count, dimension_numbers=dimension_numbers, rhs_quantization_axis=rhs_quantization_axis, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tlhs', _op._get_attr_type('Tlhs'), 'Trhs', _op._get_attr_type('Trhs'), 'Tout', _op._get_attr_type('Tout'), 'window_strides', _op.get_attr('window_strides'), 'padding', _op.get_attr('padding'), 'explicit_padding', _op.get_attr('explicit_padding'), 'lhs_dilation', _op.get_attr('lhs_dilation'), 'rhs_dilation', _op.get_attr('rhs_dilation'), 'batch_group_count', _op._get_attr_int('batch_group_count'), 'feature_group_count', _op._get_attr_int('feature_group_count'), 'dimension_numbers', _op.get_attr('dimension_numbers'), 'rhs_quantization_axis', _op._get_attr_int('rhs_quantization_axis'), 'rhs_quantization_min_val', _op._get_attr_int('rhs_quantization_min_val'), 'rhs_quantization_max_val', _op._get_attr_int('rhs_quantization_max_val'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('UniformQuantizedConvolutionHybrid', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result