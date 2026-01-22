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
def sparse_bincount_eager_fallback(indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], values: _atypes.TensorFuzzingAnnotation[TV_SparseBincount_Tidx], dense_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], size: _atypes.TensorFuzzingAnnotation[TV_SparseBincount_Tidx], weights: _atypes.TensorFuzzingAnnotation[TV_SparseBincount_T], binary_output: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_SparseBincount_T]:
    if binary_output is None:
        binary_output = False
    binary_output = _execute.make_bool(binary_output, 'binary_output')
    _attr_Tidx, _inputs_Tidx = _execute.args_to_matching_eager([values, size], ctx, [_dtypes.int32, _dtypes.int64])
    values, size = _inputs_Tidx
    _attr_T, (weights,) = _execute.args_to_matching_eager([weights], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64])
    indices = _ops.convert_to_tensor(indices, _dtypes.int64)
    dense_shape = _ops.convert_to_tensor(dense_shape, _dtypes.int64)
    _inputs_flat = [indices, values, dense_shape, size, weights]
    _attrs = ('Tidx', _attr_Tidx, 'T', _attr_T, 'binary_output', binary_output)
    _result = _execute.execute(b'SparseBincount', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseBincount', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result