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
def decode_raw_eager_fallback(bytes: _atypes.TensorFuzzingAnnotation[_atypes.String], out_type: TV_DecodeRaw_out_type, little_endian: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_DecodeRaw_out_type]:
    out_type = _execute.make_type(out_type, 'out_type')
    if little_endian is None:
        little_endian = True
    little_endian = _execute.make_bool(little_endian, 'little_endian')
    bytes = _ops.convert_to_tensor(bytes, _dtypes.string)
    _inputs_flat = [bytes]
    _attrs = ('out_type', out_type, 'little_endian', little_endian)
    _result = _execute.execute(b'DecodeRaw', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DecodeRaw', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result