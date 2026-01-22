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
def extract_jpeg_shape(contents: _atypes.TensorFuzzingAnnotation[_atypes.String], output_type: TV_ExtractJpegShape_output_type=_dtypes.int32, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ExtractJpegShape_output_type]:
    """Extract the shape information of a JPEG-encoded image.

  This op only parses the image header, so it is much faster than DecodeJpeg.

  Args:
    contents: A `Tensor` of type `string`. 0-D. The JPEG-encoded image.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
      (Optional) The output type of the operation (int32 or int64).
      Defaults to int32.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ExtractJpegShape', name, contents, 'output_type', output_type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return extract_jpeg_shape_eager_fallback(contents, output_type=output_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if output_type is None:
        output_type = _dtypes.int32
    output_type = _execute.make_type(output_type, 'output_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ExtractJpegShape', contents=contents, output_type=output_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('output_type', _op._get_attr_type('output_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ExtractJpegShape', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result