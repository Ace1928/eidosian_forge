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
def fused_pad_conv2d_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_FusedPadConv2D_T], paddings: _atypes.TensorFuzzingAnnotation[_atypes.Int32], filter: _atypes.TensorFuzzingAnnotation[TV_FusedPadConv2D_T], mode: str, strides, padding: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_FusedPadConv2D_T]:
    mode = _execute.make_str(mode, 'mode')
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'fused_pad_conv2d' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64])
    input, filter = _inputs_T
    paddings = _ops.convert_to_tensor(paddings, _dtypes.int32)
    _inputs_flat = [input, paddings, filter]
    _attrs = ('T', _attr_T, 'mode', mode, 'strides', strides, 'padding', padding)
    _result = _execute.execute(b'FusedPadConv2D', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('FusedPadConv2D', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result