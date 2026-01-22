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
def send_eager_fallback(tensor: _atypes.TensorFuzzingAnnotation[TV_Send_T], tensor_name: str, send_device: str, send_device_incarnation: int, recv_device: str, client_terminated: bool, name, ctx):
    tensor_name = _execute.make_str(tensor_name, 'tensor_name')
    send_device = _execute.make_str(send_device, 'send_device')
    send_device_incarnation = _execute.make_int(send_device_incarnation, 'send_device_incarnation')
    recv_device = _execute.make_str(recv_device, 'recv_device')
    if client_terminated is None:
        client_terminated = False
    client_terminated = _execute.make_bool(client_terminated, 'client_terminated')
    _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
    _inputs_flat = [tensor]
    _attrs = ('T', _attr_T, 'tensor_name', tensor_name, 'send_device', send_device, 'send_device_incarnation', send_device_incarnation, 'recv_device', recv_device, 'client_terminated', client_terminated)
    _result = _execute.execute(b'Send', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result