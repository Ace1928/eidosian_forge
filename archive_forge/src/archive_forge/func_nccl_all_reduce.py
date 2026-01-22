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
def nccl_all_reduce(input: _atypes.TensorFuzzingAnnotation[TV_NcclAllReduce_T], reduction: str, num_devices: int, shared_name: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_NcclAllReduce_T]:
    """Outputs a tensor containing the reduction across all input tensors.

  Outputs a tensor containing the reduction across all input tensors passed to ops
  within the same `shared_name.

  The graph should be constructed so if one op runs with shared_name value `c`,
  then `num_devices` ops will run with shared_name value `c`.  Failure to do so
  will cause the graph execution to fail to complete.

  input: the input to the reduction
  data: the value of the reduction across all `num_devices` devices.
  reduction: the reduction operation to perform.
  num_devices: The number of devices participating in this reduction.
  shared_name: Identifier that shared between ops of the same reduction.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    reduction: A `string` from: `"min", "max", "prod", "sum"`.
    num_devices: An `int`.
    shared_name: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'NcclAllReduce', name, input, 'reduction', reduction, 'num_devices', num_devices, 'shared_name', shared_name)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return nccl_all_reduce_eager_fallback(input, reduction=reduction, num_devices=num_devices, shared_name=shared_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    reduction = _execute.make_str(reduction, 'reduction')
    num_devices = _execute.make_int(num_devices, 'num_devices')
    shared_name = _execute.make_str(shared_name, 'shared_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('NcclAllReduce', input=input, reduction=reduction, num_devices=num_devices, shared_name=shared_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('reduction', _op.get_attr('reduction'), 'T', _op._get_attr_type('T'), 'num_devices', _op._get_attr_int('num_devices'), 'shared_name', _op.get_attr('shared_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('NcclAllReduce', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result