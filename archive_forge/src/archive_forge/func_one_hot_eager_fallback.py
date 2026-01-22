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
def one_hot_eager_fallback(indices: _atypes.TensorFuzzingAnnotation[TV_OneHot_TI], depth: _atypes.TensorFuzzingAnnotation[_atypes.Int32], on_value: _atypes.TensorFuzzingAnnotation[TV_OneHot_T], off_value: _atypes.TensorFuzzingAnnotation[TV_OneHot_T], axis: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_OneHot_T]:
    if axis is None:
        axis = -1
    axis = _execute.make_int(axis, 'axis')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([on_value, off_value], ctx, [])
    on_value, off_value = _inputs_T
    _attr_TI, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.uint8, _dtypes.int8, _dtypes.int32, _dtypes.int64], _dtypes.int64)
    depth = _ops.convert_to_tensor(depth, _dtypes.int32)
    _inputs_flat = [indices, depth, on_value, off_value]
    _attrs = ('axis', axis, 'T', _attr_T, 'TI', _attr_TI)
    _result = _execute.execute(b'OneHot', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('OneHot', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result