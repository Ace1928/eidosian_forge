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
def rpc_call_eager_fallback(client: _atypes.TensorFuzzingAnnotation[_atypes.Resource], method_name: _atypes.TensorFuzzingAnnotation[_atypes.String], args, timeout_in_ms: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name, ctx):
    _attr_Tin, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
    client = _ops.convert_to_tensor(client, _dtypes.resource)
    method_name = _ops.convert_to_tensor(method_name, _dtypes.string)
    timeout_in_ms = _ops.convert_to_tensor(timeout_in_ms, _dtypes.int64)
    _inputs_flat = [client, method_name] + list(args) + [timeout_in_ms]
    _attrs = ('Tin', _attr_Tin)
    _result = _execute.execute(b'RpcCall', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RpcCall', _inputs_flat, _attrs, _result)
    _result = _RpcCallOutput._make(_result)
    return _result