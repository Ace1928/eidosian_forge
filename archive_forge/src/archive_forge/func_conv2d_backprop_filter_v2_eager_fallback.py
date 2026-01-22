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
def conv2d_backprop_filter_v2_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_Conv2DBackpropFilterV2_T], filter: _atypes.TensorFuzzingAnnotation[TV_Conv2DBackpropFilterV2_T], out_backprop: _atypes.TensorFuzzingAnnotation[TV_Conv2DBackpropFilterV2_T], strides, padding: str, use_cudnn_on_gpu: bool, explicit_paddings, data_format: str, dilations, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_Conv2DBackpropFilterV2_T]:
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
    _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter, out_backprop], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64])
    input, filter, out_backprop = _inputs_T
    _inputs_flat = [input, filter, out_backprop]
    _attrs = ('T', _attr_T, 'strides', strides, 'use_cudnn_on_gpu', use_cudnn_on_gpu, 'padding', padding, 'explicit_paddings', explicit_paddings, 'data_format', data_format, 'dilations', dilations)
    _result = _execute.execute(b'Conv2DBackpropFilterV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Conv2DBackpropFilterV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result