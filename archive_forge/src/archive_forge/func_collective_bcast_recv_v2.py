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
def collective_bcast_recv_v2(group_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], group_key: _atypes.TensorFuzzingAnnotation[_atypes.Int32], instance_key: _atypes.TensorFuzzingAnnotation[_atypes.Int32], shape: _atypes.TensorFuzzingAnnotation[TV_CollectiveBcastRecvV2_Tshape], T: TV_CollectiveBcastRecvV2_T, communication_hint: str='auto', timeout_seconds: float=0, name=None) -> _atypes.TensorFuzzingAnnotation[TV_CollectiveBcastRecvV2_T]:
    """Receives a tensor value broadcast from another device.

  Args:
    group_size: A `Tensor` of type `int32`.
    group_key: A `Tensor` of type `int32`.
    instance_key: A `Tensor` of type `int32`.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    T: A `tf.DType` from: `tf.bool, tf.float32, tf.half, tf.float64, tf.int32, tf.int64`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `T`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CollectiveBcastRecvV2', name, group_size, group_key, instance_key, shape, 'T', T, 'communication_hint', communication_hint, 'timeout_seconds', timeout_seconds)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return collective_bcast_recv_v2_eager_fallback(group_size, group_key, instance_key, shape, T=T, communication_hint=communication_hint, timeout_seconds=timeout_seconds, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    T = _execute.make_type(T, 'T')
    if communication_hint is None:
        communication_hint = 'auto'
    communication_hint = _execute.make_str(communication_hint, 'communication_hint')
    if timeout_seconds is None:
        timeout_seconds = 0
    timeout_seconds = _execute.make_float(timeout_seconds, 'timeout_seconds')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CollectiveBcastRecvV2', group_size=group_size, group_key=group_key, instance_key=instance_key, shape=shape, T=T, communication_hint=communication_hint, timeout_seconds=timeout_seconds, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tshape', _op._get_attr_type('Tshape'), 'communication_hint', _op.get_attr('communication_hint'), 'timeout_seconds', _op.get_attr('timeout_seconds'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CollectiveBcastRecvV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result