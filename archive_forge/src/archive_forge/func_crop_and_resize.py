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
def crop_and_resize(image: _atypes.TensorFuzzingAnnotation[TV_CropAndResize_T], boxes: _atypes.TensorFuzzingAnnotation[_atypes.Float32], box_ind: _atypes.TensorFuzzingAnnotation[_atypes.Int32], crop_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], method: str='bilinear', extrapolation_value: float=0, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Extracts crops from the input image tensor and resizes them.

  Extracts crops from the input image tensor and resizes them using bilinear
  sampling or nearest neighbor sampling (possibly with aspect ratio change) to a
  common output size specified by `crop_size`. This is more general than the
  `crop_to_bounding_box` op which extracts a fixed size slice from the input image
  and does not allow resizing or aspect ratio change.

  Returns a tensor with `crops` from the input `image` at positions defined at the
  bounding box locations in `boxes`. The cropped boxes are all resized (with
  bilinear or nearest neighbor interpolation) to a fixed
  `size = [crop_height, crop_width]`. The result is a 4-D tensor
  `[num_boxes, crop_height, crop_width, depth]`. The resizing is corner aligned.
  In particular, if `boxes = [[0, 0, 1, 1]]`, the method will give identical
  results to using `tf.image.resize_bilinear()` or
  `tf.image.resize_nearest_neighbor()`(depends on the `method` argument) with
  `align_corners=True`.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `uint16`, `int8`, `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
      A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32`.
      A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
      specifies the coordinates of a box in the `box_ind[i]` image and is specified
      in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
      `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
      `[0, 1]` interval of normalized image height is mapped to
      `[0, image_height - 1]` in image height coordinates. We do allow `y1` > `y2`, in
      which case the sampled crop is an up-down flipped version of the original
      image. The width dimension is treated similarly. Normalized coordinates
      outside the `[0, 1]` range are allowed, in which case we use
      `extrapolation_value` to extrapolate the input image values.
    box_ind: A `Tensor` of type `int32`.
      A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
      The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
    crop_size: A `Tensor` of type `int32`.
      A 1-D tensor of 2 elements, `size = [crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the image
      content is not preserved. Both `crop_height` and `crop_width` need to be
      positive.
    method: An optional `string` from: `"bilinear", "nearest"`. Defaults to `"bilinear"`.
      A string specifying the sampling method for resizing. It can be either
      `"bilinear"` or `"nearest"` and default to `"bilinear"`. Currently two sampling
      methods are supported: Bilinear and Nearest Neighbor.
    extrapolation_value: An optional `float`. Defaults to `0`.
      Value used for extrapolation, when applicable.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CropAndResize', name, image, boxes, box_ind, crop_size, 'method', method, 'extrapolation_value', extrapolation_value)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return crop_and_resize_eager_fallback(image, boxes, box_ind, crop_size, method=method, extrapolation_value=extrapolation_value, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if method is None:
        method = 'bilinear'
    method = _execute.make_str(method, 'method')
    if extrapolation_value is None:
        extrapolation_value = 0
    extrapolation_value = _execute.make_float(extrapolation_value, 'extrapolation_value')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CropAndResize', image=image, boxes=boxes, box_ind=box_ind, crop_size=crop_size, method=method, extrapolation_value=extrapolation_value, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'method', _op.get_attr('method'), 'extrapolation_value', _op.get_attr('extrapolation_value'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CropAndResize', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result