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
def top_kv2_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_TopKV2_T], k: _atypes.TensorFuzzingAnnotation[TV_TopKV2_Tk], sorted: bool, index_type: TV_TopKV2_index_type, name, ctx):
    if sorted is None:
        sorted = True
    sorted = _execute.make_bool(sorted, 'sorted')
    if index_type is None:
        index_type = _dtypes.int32
    index_type = _execute.make_type(index_type, 'index_type')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    _attr_Tk, (k,) = _execute.args_to_matching_eager([k], ctx, [_dtypes.int16, _dtypes.int32, _dtypes.int64], _dtypes.int32)
    _inputs_flat = [input, k]
    _attrs = ('sorted', sorted, 'T', _attr_T, 'Tk', _attr_Tk, 'index_type', index_type)
    _result = _execute.execute(b'TopKV2', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TopKV2', _inputs_flat, _attrs, _result)
    _result = _TopKV2Output._make(_result)
    return _result