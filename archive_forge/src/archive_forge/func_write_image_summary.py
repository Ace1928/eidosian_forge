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
def write_image_summary(writer: _atypes.TensorFuzzingAnnotation[_atypes.Resource], step: _atypes.TensorFuzzingAnnotation[_atypes.Int64], tag: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor: _atypes.TensorFuzzingAnnotation[TV_WriteImageSummary_T], bad_color: _atypes.TensorFuzzingAnnotation[_atypes.UInt8], max_images: int=3, name=None):
    """Writes an image summary.

  Writes image `tensor` at `step` with `tag` using summary `writer`.
  `tensor` is image with shape [height, width, channels].

  Args:
    writer: A `Tensor` of type `resource`.
    step: A `Tensor` of type `int64`.
    tag: A `Tensor` of type `string`.
    tensor: A `Tensor`. Must be one of the following types: `uint8`, `float64`, `float32`, `half`.
    bad_color: A `Tensor` of type `uint8`.
    max_images: An optional `int` that is `>= 1`. Defaults to `3`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'WriteImageSummary', name, writer, step, tag, tensor, bad_color, 'max_images', max_images)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return write_image_summary_eager_fallback(writer, step, tag, tensor, bad_color, max_images=max_images, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if max_images is None:
        max_images = 3
    max_images = _execute.make_int(max_images, 'max_images')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('WriteImageSummary', writer=writer, step=step, tag=tag, tensor=tensor, bad_color=bad_color, max_images=max_images, name=name)
    return _op