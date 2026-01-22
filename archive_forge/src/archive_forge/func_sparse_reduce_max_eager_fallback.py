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
def sparse_reduce_max_eager_fallback(input_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], input_values: _atypes.TensorFuzzingAnnotation[TV_SparseReduceMax_T], input_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], reduction_axes: _atypes.TensorFuzzingAnnotation[_atypes.Int32], keep_dims: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_SparseReduceMax_T]:
    if keep_dims is None:
        keep_dims = False
    keep_dims = _execute.make_bool(keep_dims, 'keep_dims')
    _attr_T, (input_values,) = _execute.args_to_matching_eager([input_values], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    input_indices = _ops.convert_to_tensor(input_indices, _dtypes.int64)
    input_shape = _ops.convert_to_tensor(input_shape, _dtypes.int64)
    reduction_axes = _ops.convert_to_tensor(reduction_axes, _dtypes.int32)
    _inputs_flat = [input_indices, input_values, input_shape, reduction_axes]
    _attrs = ('keep_dims', keep_dims, 'T', _attr_T)
    _result = _execute.execute(b'SparseReduceMax', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseReduceMax', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result