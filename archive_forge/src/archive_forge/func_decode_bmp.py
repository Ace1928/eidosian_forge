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
def decode_bmp(contents: _atypes.TensorFuzzingAnnotation[_atypes.String], channels: int=0, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.UInt8]:
    """Decode the first frame of a BMP-encoded image to a uint8 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the BMP-encoded image.
  *   3: output an RGB image.
  *   4: output an RGBA image.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The BMP-encoded image.
    channels: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DecodeBmp', name, contents, 'channels', channels)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return decode_bmp_eager_fallback(contents, channels=channels, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if channels is None:
        channels = 0
    channels = _execute.make_int(channels, 'channels')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DecodeBmp', contents=contents, channels=channels, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('channels', _op._get_attr_int('channels'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DecodeBmp', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result