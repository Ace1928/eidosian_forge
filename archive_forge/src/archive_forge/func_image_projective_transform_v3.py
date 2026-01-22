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
def image_projective_transform_v3(images: _atypes.TensorFuzzingAnnotation[TV_ImageProjectiveTransformV3_dtype], transforms: _atypes.TensorFuzzingAnnotation[_atypes.Float32], output_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int32], fill_value: _atypes.TensorFuzzingAnnotation[_atypes.Float32], interpolation: str, fill_mode: str='CONSTANT', name=None) -> _atypes.TensorFuzzingAnnotation[TV_ImageProjectiveTransformV3_dtype]:
    """Applies the given transform to each of the images.

  If one row of `transforms` is `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps
  the *output* point `(x, y)` to a transformed *input* point
  `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
  `k = c0 x + c1 y + 1`. If the transformed point lays outside of the input
  image, the output pixel is set to fill_value.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int32`, `int64`, `half`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    transforms: A `Tensor` of type `float32`.
      2-D Tensor, `[batch, 8]` or `[1, 8]` matrix, where each row corresponds to a 3 x 3
      projective transformation matrix, with the last entry assumed to be 1. If there
      is one row, the same transformation will be applied to all images.
    output_shape: A `Tensor` of type `int32`.
      1-D Tensor [new_height, new_width].
    fill_value: A `Tensor` of type `float32`.
      float, the value to be filled when fill_mode is constant".
    interpolation: A `string`. Interpolation method, "NEAREST" or "BILINEAR".
    fill_mode: An optional `string`. Defaults to `"CONSTANT"`.
      Fill mode, "REFLECT", "WRAP", "CONSTANT", or "NEAREST".
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ImageProjectiveTransformV3', name, images, transforms, output_shape, fill_value, 'interpolation', interpolation, 'fill_mode', fill_mode)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return image_projective_transform_v3_eager_fallback(images, transforms, output_shape, fill_value, interpolation=interpolation, fill_mode=fill_mode, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    interpolation = _execute.make_str(interpolation, 'interpolation')
    if fill_mode is None:
        fill_mode = 'CONSTANT'
    fill_mode = _execute.make_str(fill_mode, 'fill_mode')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ImageProjectiveTransformV3', images=images, transforms=transforms, output_shape=output_shape, fill_value=fill_value, interpolation=interpolation, fill_mode=fill_mode, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'interpolation', _op.get_attr('interpolation'), 'fill_mode', _op.get_attr('fill_mode'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ImageProjectiveTransformV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result