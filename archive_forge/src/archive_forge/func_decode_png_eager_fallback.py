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
def decode_png_eager_fallback(contents: _atypes.TensorFuzzingAnnotation[_atypes.String], channels: int, dtype: TV_DecodePng_dtype, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_DecodePng_dtype]:
    if channels is None:
        channels = 0
    channels = _execute.make_int(channels, 'channels')
    if dtype is None:
        dtype = _dtypes.uint8
    dtype = _execute.make_type(dtype, 'dtype')
    contents = _ops.convert_to_tensor(contents, _dtypes.string)
    _inputs_flat = [contents]
    _attrs = ('channels', channels, 'dtype', dtype)
    _result = _execute.execute(b'DecodePng', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DecodePng', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result