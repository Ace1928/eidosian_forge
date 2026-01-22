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
def uniform_quantized_dot_hybrid_eager_fallback(lhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedDotHybrid_Tlhs], rhs: _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedDotHybrid_Trhs], rhs_scales: _atypes.TensorFuzzingAnnotation[_atypes.Float32], rhs_zero_points: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Tout: TV_UniformQuantizedDotHybrid_Tout, rhs_quantization_min_val: int, rhs_quantization_max_val: int, rhs_quantization_axis: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_UniformQuantizedDotHybrid_Tout]:
    Tout = _execute.make_type(Tout, 'Tout')
    rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, 'rhs_quantization_min_val')
    rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, 'rhs_quantization_max_val')
    if rhs_quantization_axis is None:
        rhs_quantization_axis = -1
    rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, 'rhs_quantization_axis')
    _attr_Tlhs, (lhs,) = _execute.args_to_matching_eager([lhs], ctx, [_dtypes.float32])
    _attr_Trhs, (rhs,) = _execute.args_to_matching_eager([rhs], ctx, [_dtypes.qint8])
    rhs_scales = _ops.convert_to_tensor(rhs_scales, _dtypes.float32)
    rhs_zero_points = _ops.convert_to_tensor(rhs_zero_points, _dtypes.int32)
    _inputs_flat = [lhs, rhs, rhs_scales, rhs_zero_points]
    _attrs = ('Tlhs', _attr_Tlhs, 'Trhs', _attr_Trhs, 'Tout', Tout, 'rhs_quantization_axis', rhs_quantization_axis, 'rhs_quantization_min_val', rhs_quantization_min_val, 'rhs_quantization_max_val', rhs_quantization_max_val)
    _result = _execute.execute(b'UniformQuantizedDotHybrid', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('UniformQuantizedDotHybrid', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result