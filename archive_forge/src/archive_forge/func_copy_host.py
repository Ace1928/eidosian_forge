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
def copy_host(input: _atypes.TensorFuzzingAnnotation[TV_CopyHost_T], tensor_name: str='', debug_ops_spec=[], name=None) -> _atypes.TensorFuzzingAnnotation[TV_CopyHost_T]:
    """Copy a tensor to host.

  Performs CPU-to-CPU deep-copying of tensor.
  N.B.: If the all downstream attached debug ops are disabled given the current
  gRPC gating status, the output will simply forward the input tensor without
  deep-copying. See the documentation of Debug* ops for more details.

  Unlike the Copy Op, this op has HostMemory constraint on its input or output.

  Args:
    input: A `Tensor`. Input tensor.
    tensor_name: An optional `string`. Defaults to `""`.
      The name of the input tensor.
    debug_ops_spec: An optional list of `strings`. Defaults to `[]`.
      A list of debug op spec (op, url, gated_grpc) for attached debug
      ops. Each element of the list has the format
      <debug_op>;<grpc_url>;<gated_grpc>, wherein gated_grpc is boolean represented
      as 0/1. E.g., "DebugIdentity;grpc://foo:3333;1",
      "DebugIdentity;file:///tmp/tfdbg_1;0".
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CopyHost', name, input, 'tensor_name', tensor_name, 'debug_ops_spec', debug_ops_spec)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return copy_host_eager_fallback(input, tensor_name=tensor_name, debug_ops_spec=debug_ops_spec, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if tensor_name is None:
        tensor_name = ''
    tensor_name = _execute.make_str(tensor_name, 'tensor_name')
    if debug_ops_spec is None:
        debug_ops_spec = []
    if not isinstance(debug_ops_spec, (list, tuple)):
        raise TypeError("Expected list for 'debug_ops_spec' argument to 'copy_host' Op, not %r." % debug_ops_spec)
    debug_ops_spec = [_execute.make_str(_s, 'debug_ops_spec') for _s in debug_ops_spec]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CopyHost', input=input, tensor_name=tensor_name, debug_ops_spec=debug_ops_spec, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'tensor_name', _op.get_attr('tensor_name'), 'debug_ops_spec', _op.get_attr('debug_ops_spec'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CopyHost', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result