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
def rpc_server_register_eager_fallback(server: _atypes.TensorFuzzingAnnotation[_atypes.Resource], method_name: _atypes.TensorFuzzingAnnotation[_atypes.String], captured_inputs, f, output_specs: str, input_specs: str, name, ctx):
    output_specs = _execute.make_str(output_specs, 'output_specs')
    if input_specs is None:
        input_specs = ''
    input_specs = _execute.make_str(input_specs, 'input_specs')
    _attr_Tin, captured_inputs = _execute.convert_to_mixed_eager_tensors(captured_inputs, ctx)
    server = _ops.convert_to_tensor(server, _dtypes.resource)
    method_name = _ops.convert_to_tensor(method_name, _dtypes.string)
    _inputs_flat = [server, method_name] + list(captured_inputs)
    _attrs = ('Tin', _attr_Tin, 'f', f, 'input_specs', input_specs, 'output_specs', output_specs)
    _result = _execute.execute(b'RpcServerRegister', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result