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
def decode_padded_raw(input_bytes: _atypes.TensorFuzzingAnnotation[_atypes.String], fixed_length: _atypes.TensorFuzzingAnnotation[_atypes.Int32], out_type: TV_DecodePaddedRaw_out_type, little_endian: bool=True, name=None) -> _atypes.TensorFuzzingAnnotation[TV_DecodePaddedRaw_out_type]:
    """Reinterpret the bytes of a string as a vector of numbers.

  Args:
    input_bytes: A `Tensor` of type `string`. Tensor of string to be decoded.
    fixed_length: A `Tensor` of type `int32`.
      Length in bytes for each element of the decoded output. Must be a multiple
      of the size of the output type.
    out_type: A `tf.DType` from: `tf.half, tf.float32, tf.float64, tf.int32, tf.uint16, tf.uint8, tf.int16, tf.int8, tf.int64, tf.bfloat16`.
    little_endian: An optional `bool`. Defaults to `True`.
      Whether the input `input_bytes` is in little-endian order. Ignored for
      `out_type` values that are stored in a single byte, like `uint8`
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DecodePaddedRaw', name, input_bytes, fixed_length, 'out_type', out_type, 'little_endian', little_endian)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return decode_padded_raw_eager_fallback(input_bytes, fixed_length, out_type=out_type, little_endian=little_endian, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    out_type = _execute.make_type(out_type, 'out_type')
    if little_endian is None:
        little_endian = True
    little_endian = _execute.make_bool(little_endian, 'little_endian')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DecodePaddedRaw', input_bytes=input_bytes, fixed_length=fixed_length, out_type=out_type, little_endian=little_endian, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('out_type', _op._get_attr_type('out_type'), 'little_endian', _op._get_attr_bool('little_endian'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DecodePaddedRaw', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result