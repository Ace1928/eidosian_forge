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
def image_projective_transform_v3_eager_fallback(images: _atypes.TensorFuzzingAnnotation[TV_ImageProjectiveTransformV3_dtype], transforms: _atypes.TensorFuzzingAnnotation[_atypes.Float32], output_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int32], fill_value: _atypes.TensorFuzzingAnnotation[_atypes.Float32], interpolation: str, fill_mode: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_ImageProjectiveTransformV3_dtype]:
    interpolation = _execute.make_str(interpolation, 'interpolation')
    if fill_mode is None:
        fill_mode = 'CONSTANT'
    fill_mode = _execute.make_str(fill_mode, 'fill_mode')
    _attr_dtype, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.uint8, _dtypes.int32, _dtypes.int64, _dtypes.half, _dtypes.float32, _dtypes.float64])
    transforms = _ops.convert_to_tensor(transforms, _dtypes.float32)
    output_shape = _ops.convert_to_tensor(output_shape, _dtypes.int32)
    fill_value = _ops.convert_to_tensor(fill_value, _dtypes.float32)
    _inputs_flat = [images, transforms, output_shape, fill_value]
    _attrs = ('dtype', _attr_dtype, 'interpolation', interpolation, 'fill_mode', fill_mode)
    _result = _execute.execute(b'ImageProjectiveTransformV3', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ImageProjectiveTransformV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result