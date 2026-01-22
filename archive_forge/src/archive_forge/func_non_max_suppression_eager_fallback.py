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
def non_max_suppression_eager_fallback(boxes: _atypes.TensorFuzzingAnnotation[_atypes.Float32], scores: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_output_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], iou_threshold: float, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Int32]:
    if iou_threshold is None:
        iou_threshold = 0.5
    iou_threshold = _execute.make_float(iou_threshold, 'iou_threshold')
    boxes = _ops.convert_to_tensor(boxes, _dtypes.float32)
    scores = _ops.convert_to_tensor(scores, _dtypes.float32)
    max_output_size = _ops.convert_to_tensor(max_output_size, _dtypes.int32)
    _inputs_flat = [boxes, scores, max_output_size]
    _attrs = ('iou_threshold', iou_threshold)
    _result = _execute.execute(b'NonMaxSuppression', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('NonMaxSuppression', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result