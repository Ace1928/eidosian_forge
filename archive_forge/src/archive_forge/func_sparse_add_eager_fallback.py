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
def sparse_add_eager_fallback(a_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], a_values: _atypes.TensorFuzzingAnnotation[TV_SparseAdd_T], a_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], b_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], b_values: _atypes.TensorFuzzingAnnotation[TV_SparseAdd_T], b_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], thresh: _atypes.TensorFuzzingAnnotation[TV_SparseAdd_Treal], name, ctx):
    _attr_T, _inputs_T = _execute.args_to_matching_eager([a_values, b_values], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    a_values, b_values = _inputs_T
    _attr_Treal, (thresh,) = _execute.args_to_matching_eager([thresh], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    a_indices = _ops.convert_to_tensor(a_indices, _dtypes.int64)
    a_shape = _ops.convert_to_tensor(a_shape, _dtypes.int64)
    b_indices = _ops.convert_to_tensor(b_indices, _dtypes.int64)
    b_shape = _ops.convert_to_tensor(b_shape, _dtypes.int64)
    _inputs_flat = [a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh]
    _attrs = ('T', _attr_T, 'Treal', _attr_Treal)
    _result = _execute.execute(b'SparseAdd', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseAdd', _inputs_flat, _attrs, _result)
    _result = _SparseAddOutput._make(_result)
    return _result