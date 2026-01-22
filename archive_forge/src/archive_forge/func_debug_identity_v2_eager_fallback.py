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
def debug_identity_v2_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_DebugIdentityV2_T], tfdbg_context_id: str, op_name: str, output_slot: int, tensor_debug_mode: int, debug_urls, circular_buffer_size: int, tfdbg_run_id: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_DebugIdentityV2_T]:
    if tfdbg_context_id is None:
        tfdbg_context_id = ''
    tfdbg_context_id = _execute.make_str(tfdbg_context_id, 'tfdbg_context_id')
    if op_name is None:
        op_name = ''
    op_name = _execute.make_str(op_name, 'op_name')
    if output_slot is None:
        output_slot = -1
    output_slot = _execute.make_int(output_slot, 'output_slot')
    if tensor_debug_mode is None:
        tensor_debug_mode = -1
    tensor_debug_mode = _execute.make_int(tensor_debug_mode, 'tensor_debug_mode')
    if debug_urls is None:
        debug_urls = []
    if not isinstance(debug_urls, (list, tuple)):
        raise TypeError("Expected list for 'debug_urls' argument to 'debug_identity_v2' Op, not %r." % debug_urls)
    debug_urls = [_execute.make_str(_s, 'debug_urls') for _s in debug_urls]
    if circular_buffer_size is None:
        circular_buffer_size = 1000
    circular_buffer_size = _execute.make_int(circular_buffer_size, 'circular_buffer_size')
    if tfdbg_run_id is None:
        tfdbg_run_id = ''
    tfdbg_run_id = _execute.make_str(tfdbg_run_id, 'tfdbg_run_id')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
    _inputs_flat = [input]
    _attrs = ('T', _attr_T, 'tfdbg_context_id', tfdbg_context_id, 'op_name', op_name, 'output_slot', output_slot, 'tensor_debug_mode', tensor_debug_mode, 'debug_urls', debug_urls, 'circular_buffer_size', circular_buffer_size, 'tfdbg_run_id', tfdbg_run_id)
    _result = _execute.execute(b'DebugIdentityV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DebugIdentityV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result