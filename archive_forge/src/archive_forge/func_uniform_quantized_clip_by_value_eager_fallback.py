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
def uniform_quantized_clip_by_value_eager_fallback(operand: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedClipByValue_T], min: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedClipByValue_T], max: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedClipByValue_T], scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], quantization_min_val: int, quantization_max_val: int, quantization_axis: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedClipByValue_T]:
    quantization_min_val = _execute.make_int(quantization_min_val, 'quantization_min_val')
    quantization_max_val = _execute.make_int(quantization_max_val, 'quantization_max_val')
    if quantization_axis is None:
        quantization_axis = -1
    quantization_axis = _execute.make_int(quantization_axis, 'quantization_axis')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([operand, min, max], ctx, [_dtypes.qint32])
    operand, min, max = _inputs_T
    scales = _ops.convert_to_tensor(scales, _dtypes.float32)
    zero_points = _ops.convert_to_tensor(zero_points, _dtypes.int32)
    _inputs_flat = [operand, min, max, scales, zero_points]
    _attrs = ('T', _attr_T, 'quantization_axis', quantization_axis, 'quantization_min_val', quantization_min_val, 'quantization_max_val', quantization_max_val)
    _result = _execute.execute(b'UniformQuantizedClipByValue', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('UniformQuantizedClipByValue', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result