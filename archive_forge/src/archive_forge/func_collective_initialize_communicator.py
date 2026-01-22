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
def collective_initialize_communicator(group_key: _atypes.TensorFuzzingAnnotation[_atypes.Int32], rank: _atypes.TensorFuzzingAnnotation[_atypes.Int32], group_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], communication_hint: str='auto', timeout_seconds: float=0, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    """Initializes a group for collective operations.

  Args:
    group_key: A `Tensor` of type `int32`.
    rank: A `Tensor` of type `int32`.
    group_size: A `Tensor` of type `int32`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CollectiveInitializeCommunicator', name, group_key, rank, group_size, 'communication_hint', communication_hint, 'timeout_seconds', timeout_seconds)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return collective_initialize_communicator_eager_fallback(group_key, rank, group_size, communication_hint=communication_hint, timeout_seconds=timeout_seconds, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if communication_hint is None:
        communication_hint = 'auto'
    communication_hint = _execute.make_str(communication_hint, 'communication_hint')
    if timeout_seconds is None:
        timeout_seconds = 0
    timeout_seconds = _execute.make_float(timeout_seconds, 'timeout_seconds')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CollectiveInitializeCommunicator', group_key=group_key, rank=rank, group_size=group_size, communication_hint=communication_hint, timeout_seconds=timeout_seconds, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('communication_hint', _op.get_attr('communication_hint'), 'timeout_seconds', _op.get_attr('timeout_seconds'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CollectiveInitializeCommunicator', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result