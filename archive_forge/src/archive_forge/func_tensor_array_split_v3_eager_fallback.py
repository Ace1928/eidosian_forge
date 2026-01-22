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
def tensor_array_split_v3_eager_fallback(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], value: _atypes.TensorFuzzingAnnotation[TV_TensorArraySplitV3_T], lengths: _atypes.TensorFuzzingAnnotation[_atypes.Int64], flow_in: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
    handle = _ops.convert_to_tensor(handle, _dtypes.resource)
    lengths = _ops.convert_to_tensor(lengths, _dtypes.int64)
    flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
    _inputs_flat = [handle, value, lengths, flow_in]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'TensorArraySplitV3', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TensorArraySplitV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result