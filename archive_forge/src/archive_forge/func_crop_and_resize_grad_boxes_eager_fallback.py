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
def crop_and_resize_grad_boxes_eager_fallback(grads: _atypes.TensorFuzzingAnnotation[_atypes.Float32], image: _atypes.TensorFuzzingAnnotation[TV_CropAndResizeGradBoxes_T], boxes: _atypes.TensorFuzzingAnnotation[_atypes.Float32], box_ind: _atypes.TensorFuzzingAnnotation[_atypes.Int32], method: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    if method is None:
        method = 'bilinear'
    method = _execute.make_str(method, 'method')
    _attr_T, (image,) = _execute.args_to_matching_eager([image], ctx, [_dtypes.uint8, _dtypes.uint16, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.half, _dtypes.float32, _dtypes.float64])
    grads = _ops.convert_to_tensor(grads, _dtypes.float32)
    boxes = _ops.convert_to_tensor(boxes, _dtypes.float32)
    box_ind = _ops.convert_to_tensor(box_ind, _dtypes.int32)
    _inputs_flat = [grads, image, boxes, box_ind]
    _attrs = ('T', _attr_T, 'method', method)
    _result = _execute.execute(b'CropAndResizeGradBoxes', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CropAndResizeGradBoxes', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result