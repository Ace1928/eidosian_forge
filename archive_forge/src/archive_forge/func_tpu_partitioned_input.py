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
def tpu_partitioned_input(inputs: List[_atypes.TensorFuzzingAnnotation[TV_TPUPartitionedInput_T]], partition_dim: int=0, name=None) -> _atypes.TensorFuzzingAnnotation[TV_TPUPartitionedInput_T]:
    """An op that groups a list of partitioned inputs together. This op

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type.
      A list of partitioned inputs which must have the same shape.
    partition_dim: An optional `int`. Defaults to `0`.
      An integer describles which dimension is partitioned. -1 means
      those inputs are replicated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TPUPartitionedInput', name, inputs, 'partition_dim', partition_dim)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tpu_partitioned_input_eager_fallback(inputs, partition_dim=partition_dim, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Expected list for 'inputs' argument to 'tpu_partitioned_input' Op, not %r." % inputs)
    _attr_N = len(inputs)
    if partition_dim is None:
        partition_dim = 0
    partition_dim = _execute.make_int(partition_dim, 'partition_dim')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TPUPartitionedInput', inputs=inputs, partition_dim=partition_dim, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('N', _op._get_attr_int('N'), 'T', _op._get_attr_type('T'), 'partition_dim', _op._get_attr_int('partition_dim'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TPUPartitionedInput', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result