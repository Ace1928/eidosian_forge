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
def tpu_partitioned_output_v2(inputs: _atypes.TensorFuzzingAnnotation[TV_TPUPartitionedOutputV2_T], num_splits: int, partition_dims, name=None):
    """An op that demultiplexes a tensor to be sharded by XLA to a list of partitioned

  outputs outside the XLA computation. Supports ND sharding.

  Args:
    inputs: A `Tensor`.
      A tensor which represents the full shape of partitioned tensors.
    num_splits: An `int` that is `>= 1`.
    partition_dims: A list of `ints`.
      A list of integers describing how each dimension is partitioned. Emptiness
      indicates the inputs are replicated.
    name: A name for the operation (optional).

  Returns:
    A list of `num_splits` `Tensor` objects with the same type as `inputs`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TPUPartitionedOutputV2', name, inputs, 'num_splits', num_splits, 'partition_dims', partition_dims)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tpu_partitioned_output_v2_eager_fallback(inputs, num_splits=num_splits, partition_dims=partition_dims, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    num_splits = _execute.make_int(num_splits, 'num_splits')
    if not isinstance(partition_dims, (list, tuple)):
        raise TypeError("Expected list for 'partition_dims' argument to 'tpu_partitioned_output_v2' Op, not %r." % partition_dims)
    partition_dims = [_execute.make_int(_i, 'partition_dims') for _i in partition_dims]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TPUPartitionedOutputV2', inputs=inputs, num_splits=num_splits, partition_dims=partition_dims, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'num_splits', _op._get_attr_int('num_splits'), 'partition_dims', _op.get_attr('partition_dims'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TPUPartitionedOutputV2', _inputs_flat, _attrs, _result)
    return _result