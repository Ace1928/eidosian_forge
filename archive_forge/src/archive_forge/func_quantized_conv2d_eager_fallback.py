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
def quantized_conv2d_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_QuantizedConv2D_Tinput], filter: _atypes.TensorFuzzingAnnotation[TV_QuantizedConv2D_Tfilter], min_input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_filter: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_filter: _atypes.TensorFuzzingAnnotation[_atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2D_out_type, dilations, name, ctx):
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'quantized_conv2d' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if out_type is None:
        out_type = _dtypes.qint32
    out_type = _execute.make_type(out_type, 'out_type')
    if dilations is None:
        dilations = [1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
        raise TypeError("Expected list for 'dilations' argument to 'quantized_conv2d' Op, not %r." % dilations)
    dilations = [_execute.make_int(_i, 'dilations') for _i in dilations]
    _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
    max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
    min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
    max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
    _inputs_flat = [input, filter, min_input, max_input, min_filter, max_filter]
    _attrs = ('Tinput', _attr_Tinput, 'Tfilter', _attr_Tfilter, 'out_type', out_type, 'strides', strides, 'padding', padding, 'dilations', dilations)
    _result = _execute.execute(b'QuantizedConv2D', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizedConv2D', _inputs_flat, _attrs, _result)
    _result = _QuantizedConv2DOutput._make(_result)
    return _result