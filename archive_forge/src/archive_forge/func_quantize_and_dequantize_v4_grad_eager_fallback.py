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
def quantize_and_dequantize_v4_grad_eager_fallback(gradients: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV4Grad_T], input: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV4Grad_T], input_min: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV4Grad_T], input_max: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV4Grad_T], axis: int, name, ctx):
    if axis is None:
        axis = -1
    axis = _execute.make_int(axis, 'axis')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([gradients, input, input_min, input_max], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64])
    gradients, input, input_min, input_max = _inputs_T
    _inputs_flat = [gradients, input, input_min, input_max]
    _attrs = ('T', _attr_T, 'axis', axis)
    _result = _execute.execute(b'QuantizeAndDequantizeV4Grad', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizeAndDequantizeV4Grad', _inputs_flat, _attrs, _result)
    _result = _QuantizeAndDequantizeV4GradOutput._make(_result)
    return _result