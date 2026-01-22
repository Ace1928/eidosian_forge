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
def quantized_resize_bilinear_eager_fallback(images: _atypes.TensorFuzzingAnnotation[TV_QuantizedResizeBilinear_T], size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], align_corners: bool, half_pixel_centers: bool, name, ctx):
    if align_corners is None:
        align_corners = False
    align_corners = _execute.make_bool(align_corners, 'align_corners')
    if half_pixel_centers is None:
        half_pixel_centers = False
    half_pixel_centers = _execute.make_bool(half_pixel_centers, 'half_pixel_centers')
    _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.quint8, _dtypes.qint32, _dtypes.float32])
    size = _ops.convert_to_tensor(size, _dtypes.int32)
    min = _ops.convert_to_tensor(min, _dtypes.float32)
    max = _ops.convert_to_tensor(max, _dtypes.float32)
    _inputs_flat = [images, size, min, max]
    _attrs = ('T', _attr_T, 'align_corners', align_corners, 'half_pixel_centers', half_pixel_centers)
    _result = _execute.execute(b'QuantizedResizeBilinear', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizedResizeBilinear', _inputs_flat, _attrs, _result)
    _result = _QuantizedResizeBilinearOutput._make(_result)
    return _result