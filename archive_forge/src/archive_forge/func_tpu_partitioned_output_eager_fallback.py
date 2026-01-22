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
def tpu_partitioned_output_eager_fallback(inputs: _atypes.TensorFuzzingAnnotation[TV_TPUPartitionedOutput_T], num_splits: int, partition_dim: int, name, ctx):
    num_splits = _execute.make_int(num_splits, 'num_splits')
    if partition_dim is None:
        partition_dim = 0
    partition_dim = _execute.make_int(partition_dim, 'partition_dim')
    _attr_T, (inputs,) = _execute.args_to_matching_eager([inputs], ctx, [])
    _inputs_flat = [inputs]
    _attrs = ('T', _attr_T, 'num_splits', num_splits, 'partition_dim', partition_dim)
    _result = _execute.execute(b'TPUPartitionedOutput', num_splits, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TPUPartitionedOutput', _inputs_flat, _attrs, _result)
    return _result