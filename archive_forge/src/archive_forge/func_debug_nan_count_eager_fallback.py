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
def debug_nan_count_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_DebugNanCount_T], device_name: str, tensor_name: str, debug_urls, gated_grpc: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    if device_name is None:
        device_name = ''
    device_name = _execute.make_str(device_name, 'device_name')
    if tensor_name is None:
        tensor_name = ''
    tensor_name = _execute.make_str(tensor_name, 'tensor_name')
    if debug_urls is None:
        debug_urls = []
    if not isinstance(debug_urls, (list, tuple)):
        raise TypeError("Expected list for 'debug_urls' argument to 'debug_nan_count' Op, not %r." % debug_urls)
    debug_urls = [_execute.make_str(_s, 'debug_urls') for _s in debug_urls]
    if gated_grpc is None:
        gated_grpc = False
    gated_grpc = _execute.make_bool(gated_grpc, 'gated_grpc')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
    _inputs_flat = [input]
    _attrs = ('T', _attr_T, 'device_name', device_name, 'tensor_name', tensor_name, 'debug_urls', debug_urls, 'gated_grpc', gated_grpc)
    _result = _execute.execute(b'DebugNanCount', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DebugNanCount', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result