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
def xla_gather_eager_fallback(operand: _atypes.TensorFuzzingAnnotation[TV_XlaGather_T], start_indices: _atypes.TensorFuzzingAnnotation[TV_XlaGather_Tindices], slice_sizes: _atypes.TensorFuzzingAnnotation[TV_XlaGather_Tindices], dimension_numbers: str, indices_are_sorted: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_XlaGather_T]:
    dimension_numbers = _execute.make_str(dimension_numbers, 'dimension_numbers')
    indices_are_sorted = _execute.make_bool(indices_are_sorted, 'indices_are_sorted')
    _attr_T, (operand,) = _execute.args_to_matching_eager([operand], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.bool])
    _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([start_indices, slice_sizes], ctx, [_dtypes.int32, _dtypes.int64])
    start_indices, slice_sizes = _inputs_Tindices
    _inputs_flat = [operand, start_indices, slice_sizes]
    _attrs = ('dimension_numbers', dimension_numbers, 'indices_are_sorted', indices_are_sorted, 'T', _attr_T, 'Tindices', _attr_Tindices)
    _result = _execute.execute(b'XlaGather', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaGather', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result