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
def debug_identity_v2(input: _atypes.TensorFuzzingAnnotation[TV_DebugIdentityV2_T], tfdbg_context_id: str='', op_name: str='', output_slot: int=-1, tensor_debug_mode: int=-1, debug_urls=[], circular_buffer_size: int=1000, tfdbg_run_id: str='', name=None) -> _atypes.TensorFuzzingAnnotation[TV_DebugIdentityV2_T]:
    """Debug Identity V2 Op.

  Provides an identity mapping from input to output, while writing the content of
  the input tensor by calling DebugEventsWriter.

  The semantics of the input tensor depends on tensor_debug_mode. In typical
  usage, the input tensor comes directly from the user computation only when
  graph_debug_mode is FULL_TENSOR (see protobuf/debug_event.proto for a
  list of all the possible values of graph_debug_mode). For the other debug modes,
  the input tensor should be produced by an additional op or subgraph that
  computes summary information about one or more tensors.

  Args:
    input: A `Tensor`. Input tensor, non-Reference type
    tfdbg_context_id: An optional `string`. Defaults to `""`.
      A tfdbg-generated ID for the context that the op belongs to,
        e.g., a concrete compiled tf.function.
    op_name: An optional `string`. Defaults to `""`.
      Optional. Name of the op that the debug op is concerned with.
        Used only for single-tensor trace.
    output_slot: An optional `int`. Defaults to `-1`.
      Optional. Output slot index of the tensor that the debug op
        is concerned with. Used only for single-tensor trace.
    tensor_debug_mode: An optional `int`. Defaults to `-1`.
      TensorDebugMode enum value. See debug_event.proto for details.
    debug_urls: An optional list of `strings`. Defaults to `[]`.
      List of URLs to debug targets, e.g., file:///foo/tfdbg_dump.
    circular_buffer_size: An optional `int`. Defaults to `1000`.
    tfdbg_run_id: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DebugIdentityV2', name, input, 'tfdbg_context_id', tfdbg_context_id, 'op_name', op_name, 'output_slot', output_slot, 'tensor_debug_mode', tensor_debug_mode, 'debug_urls', debug_urls, 'circular_buffer_size', circular_buffer_size, 'tfdbg_run_id', tfdbg_run_id)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return debug_identity_v2_eager_fallback(input, tfdbg_context_id=tfdbg_context_id, op_name=op_name, output_slot=output_slot, tensor_debug_mode=tensor_debug_mode, debug_urls=debug_urls, circular_buffer_size=circular_buffer_size, tfdbg_run_id=tfdbg_run_id, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
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
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DebugIdentityV2', input=input, tfdbg_context_id=tfdbg_context_id, op_name=op_name, output_slot=output_slot, tensor_debug_mode=tensor_debug_mode, debug_urls=debug_urls, circular_buffer_size=circular_buffer_size, tfdbg_run_id=tfdbg_run_id, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'tfdbg_context_id', _op.get_attr('tfdbg_context_id'), 'op_name', _op.get_attr('op_name'), 'output_slot', _op._get_attr_int('output_slot'), 'tensor_debug_mode', _op._get_attr_int('tensor_debug_mode'), 'debug_urls', _op.get_attr('debug_urls'), 'circular_buffer_size', _op._get_attr_int('circular_buffer_size'), 'tfdbg_run_id', _op.get_attr('tfdbg_run_id'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DebugIdentityV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result