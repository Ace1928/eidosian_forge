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
def sparse_segment_sqrt_n_with_num_segments_eager_fallback(data: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNWithNumSegments_T], indices: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNWithNumSegments_Tidx], segment_ids: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNWithNumSegments_Tsegmentids], num_segments: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNWithNumSegments_Tnumsegments], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNWithNumSegments_T]:
    _attr_T, (data,) = _execute.args_to_matching_eager([data], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64])
    _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _attr_Tnumsegments, (num_segments,) = _execute.args_to_matching_eager([num_segments], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _inputs_flat = [data, indices, segment_ids, num_segments]
    _attrs = ('T', _attr_T, 'Tidx', _attr_Tidx, 'Tnumsegments', _attr_Tnumsegments, 'Tsegmentids', _attr_Tsegmentids)
    _result = _execute.execute(b'SparseSegmentSqrtNWithNumSegments', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseSegmentSqrtNWithNumSegments', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result