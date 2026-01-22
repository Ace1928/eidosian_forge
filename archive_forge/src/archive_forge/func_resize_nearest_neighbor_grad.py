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
def resize_nearest_neighbor_grad(grads: _atypes.TensorFuzzingAnnotation[TV_ResizeNearestNeighborGrad_T], size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], align_corners: bool=False, half_pixel_centers: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ResizeNearestNeighborGrad_T]:
    """Computes the gradient of nearest neighbor interpolation.

  Args:
    grads: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `half`, `float32`, `float64`, `bfloat16`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The
      original input size.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and grad tensors are
      aligned. Defaults to false.
    half_pixel_centers: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grads`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResizeNearestNeighborGrad', name, grads, size, 'align_corners', align_corners, 'half_pixel_centers', half_pixel_centers)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resize_nearest_neighbor_grad_eager_fallback(grads, size, align_corners=align_corners, half_pixel_centers=half_pixel_centers, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if align_corners is None:
        align_corners = False
    align_corners = _execute.make_bool(align_corners, 'align_corners')
    if half_pixel_centers is None:
        half_pixel_centers = False
    half_pixel_centers = _execute.make_bool(half_pixel_centers, 'half_pixel_centers')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResizeNearestNeighborGrad', grads=grads, size=size, align_corners=align_corners, half_pixel_centers=half_pixel_centers, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'align_corners', _op._get_attr_bool('align_corners'), 'half_pixel_centers', _op._get_attr_bool('half_pixel_centers'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ResizeNearestNeighborGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result