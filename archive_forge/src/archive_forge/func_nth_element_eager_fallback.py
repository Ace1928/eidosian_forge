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
def nth_element_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_NthElement_T], n: _atypes.TensorFuzzingAnnotation[_atypes.Int32], reverse: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_NthElement_T]:
    if reverse is None:
        reverse = False
    reverse = _execute.make_bool(reverse, 'reverse')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    n = _ops.convert_to_tensor(n, _dtypes.int32)
    _inputs_flat = [input, n]
    _attrs = ('reverse', reverse, 'T', _attr_T)
    _result = _execute.execute(b'NthElement', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('NthElement', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result