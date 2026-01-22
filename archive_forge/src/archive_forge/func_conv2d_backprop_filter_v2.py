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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('conv2d_backprop_filter_v2')
def conv2d_backprop_filter_v2(input: _atypes.TensorFuzzingAnnotation[TV_Conv2DBackpropFilterV2_T], filter: _atypes.TensorFuzzingAnnotation[TV_Conv2DBackpropFilterV2_T], out_backprop: _atypes.TensorFuzzingAnnotation[TV_Conv2DBackpropFilterV2_T], strides, padding: str, use_cudnn_on_gpu: bool=True, explicit_paddings=[], data_format: str='NHWC', dilations=[1, 1, 1, 1], name=None) -> _atypes.TensorFuzzingAnnotation[TV_Conv2DBackpropFilterV2_T]:
    """Computes the gradients of convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[filter_height, filter_width, in_channels, out_channels]`.
      Only shape of tensor is used.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `"SAME", "VALID", "EXPLICIT"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Conv2DBackpropFilterV2', name, input, filter, out_backprop, 'strides', strides, 'use_cudnn_on_gpu', use_cudnn_on_gpu, 'padding', padding, 'explicit_paddings', explicit_paddings, 'data_format', data_format, 'dilations', dilations)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_conv2d_backprop_filter_v2((input, filter, out_backprop, strides, padding, use_cudnn_on_gpu, explicit_paddings, data_format, dilations, name), None)
            if _result is not NotImplemented:
                return _result
            return conv2d_backprop_filter_v2_eager_fallback(input, filter, out_backprop, strides=strides, use_cudnn_on_gpu=use_cudnn_on_gpu, padding=padding, explicit_paddings=explicit_paddings, data_format=data_format, dilations=dilations, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(conv2d_backprop_filter_v2, (), dict(input=input, filter=filter, out_backprop=out_backprop, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, explicit_paddings=explicit_paddings, data_format=data_format, dilations=dilations, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_conv2d_backprop_filter_v2((input, filter, out_backprop, strides, padding, use_cudnn_on_gpu, explicit_paddings, data_format, dilations, name), None)
        if _result is not NotImplemented:
            return _result
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'conv2d_backprop_filter_v2' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if use_cudnn_on_gpu is None:
        use_cudnn_on_gpu = True
    use_cudnn_on_gpu = _execute.make_bool(use_cudnn_on_gpu, 'use_cudnn_on_gpu')
    if explicit_paddings is None:
        explicit_paddings = []
    if not isinstance(explicit_paddings, (list, tuple)):
        raise TypeError("Expected list for 'explicit_paddings' argument to 'conv2d_backprop_filter_v2' Op, not %r." % explicit_paddings)
    explicit_paddings = [_execute.make_int(_i, 'explicit_paddings') for _i in explicit_paddings]
    if data_format is None:
        data_format = 'NHWC'
    data_format = _execute.make_str(data_format, 'data_format')
    if dilations is None:
        dilations = [1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
        raise TypeError("Expected list for 'dilations' argument to 'conv2d_backprop_filter_v2' Op, not %r." % dilations)
    dilations = [_execute.make_int(_i, 'dilations') for _i in dilations]
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('Conv2DBackpropFilterV2', input=input, filter=filter, out_backprop=out_backprop, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, explicit_paddings=explicit_paddings, data_format=data_format, dilations=dilations, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(conv2d_backprop_filter_v2, (), dict(input=input, filter=filter, out_backprop=out_backprop, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, explicit_paddings=explicit_paddings, data_format=data_format, dilations=dilations, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'strides', _op.get_attr('strides'), 'use_cudnn_on_gpu', _op._get_attr_bool('use_cudnn_on_gpu'), 'padding', _op.get_attr('padding'), 'explicit_paddings', _op.get_attr('explicit_paddings'), 'data_format', _op.get_attr('data_format'), 'dilations', _op.get_attr('dilations'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Conv2DBackpropFilterV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result