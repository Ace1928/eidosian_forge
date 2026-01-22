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
def non_max_suppression_with_overlaps_eager_fallback(overlaps: _atypes.TensorFuzzingAnnotation[_atypes.Float32], scores: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_output_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], overlap_threshold: _atypes.TensorFuzzingAnnotation[_atypes.Float32], score_threshold: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Int32]:
    overlaps = _ops.convert_to_tensor(overlaps, _dtypes.float32)
    scores = _ops.convert_to_tensor(scores, _dtypes.float32)
    max_output_size = _ops.convert_to_tensor(max_output_size, _dtypes.int32)
    overlap_threshold = _ops.convert_to_tensor(overlap_threshold, _dtypes.float32)
    score_threshold = _ops.convert_to_tensor(score_threshold, _dtypes.float32)
    _inputs_flat = [overlaps, scores, max_output_size, overlap_threshold, score_threshold]
    _attrs = None
    _result = _execute.execute(b'NonMaxSuppressionWithOverlaps', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('NonMaxSuppressionWithOverlaps', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result