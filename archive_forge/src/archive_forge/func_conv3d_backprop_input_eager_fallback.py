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
def conv3d_backprop_input_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_Conv3DBackpropInput_T], filter: _atypes.TensorFuzzingAnnotation[TV_Conv3DBackpropInput_T], out_backprop: _atypes.TensorFuzzingAnnotation[TV_Conv3DBackpropInput_T], strides, padding: str, dilations, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_Conv3DBackpropInput_T]:
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'conv3d_backprop_input' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if dilations is None:
        dilations = [1, 1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
        raise TypeError("Expected list for 'dilations' argument to 'conv3d_backprop_input' Op, not %r." % dilations)
    dilations = [_execute.make_int(_i, 'dilations') for _i in dilations]
    _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter, out_backprop], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64])
    input, filter, out_backprop = _inputs_T
    _inputs_flat = [input, filter, out_backprop]
    _attrs = ('T', _attr_T, 'strides', strides, 'padding', padding, 'dilations', dilations)
    _result = _execute.execute(b'Conv3DBackpropInput', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Conv3DBackpropInput', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result