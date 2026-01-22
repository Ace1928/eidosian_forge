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
def nccl_broadcast(input: _atypes.TensorFuzzingAnnotation[TV_NcclBroadcast_T], shape, name=None) -> _atypes.TensorFuzzingAnnotation[TV_NcclBroadcast_T]:
    """Sends `input` to all devices that are connected to the output.

  Sends `input` to all devices that are connected to the output.

  The graph should be constructed so that all ops connected to the output have a
  valid device assignment, and the op itself is assigned one of these devices.

  input: The input to the broadcast.
  output: The same as input.
  shape: The shape of the input tensor.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    shape: A `tf.TensorShape` or list of `ints`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'NcclBroadcast', name, input, 'shape', shape)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return nccl_broadcast_eager_fallback(input, shape=shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    shape = _execute.make_shape(shape, 'shape')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('NcclBroadcast', input=input, shape=shape, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'shape', _op.get_attr('shape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('NcclBroadcast', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result