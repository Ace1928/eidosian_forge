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
def write_image_summary_eager_fallback(writer: _atypes.TensorFuzzingAnnotation[_atypes.Resource], step: _atypes.TensorFuzzingAnnotation[_atypes.Int64], tag: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor: _atypes.TensorFuzzingAnnotation[TV_WriteImageSummary_T], bad_color: _atypes.TensorFuzzingAnnotation[_atypes.UInt8], max_images: int, name, ctx):
    if max_images is None:
        max_images = 3
    max_images = _execute.make_int(max_images, 'max_images')
    _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [_dtypes.uint8, _dtypes.float64, _dtypes.float32, _dtypes.half], _dtypes.float32)
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    step = _ops.convert_to_tensor(step, _dtypes.int64)
    tag = _ops.convert_to_tensor(tag, _dtypes.string)
    bad_color = _ops.convert_to_tensor(bad_color, _dtypes.uint8)
    _inputs_flat = [writer, step, tag, tensor, bad_color]
    _attrs = ('max_images', max_images, 'T', _attr_T)
    _result = _execute.execute(b'WriteImageSummary', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result