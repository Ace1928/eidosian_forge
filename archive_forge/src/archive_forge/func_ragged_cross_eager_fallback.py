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
def ragged_cross_eager_fallback(ragged_values, ragged_row_splits, sparse_indices: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], sparse_values, sparse_shape: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], dense_inputs, input_order: str, hashed_output: bool, num_buckets: int, hash_key: int, out_values_type: TV_RaggedCross_out_values_type, out_row_splits_type: TV_RaggedCross_out_row_splits_type, name, ctx):
    if not isinstance(sparse_indices, (list, tuple)):
        raise TypeError("Expected list for 'sparse_indices' argument to 'ragged_cross' Op, not %r." % sparse_indices)
    _attr_Nsparse = len(sparse_indices)
    if not isinstance(sparse_shape, (list, tuple)):
        raise TypeError("Expected list for 'sparse_shape' argument to 'ragged_cross' Op, not %r." % sparse_shape)
    if len(sparse_shape) != _attr_Nsparse:
        raise ValueError("List argument 'sparse_shape' to 'ragged_cross' Op with length %d must match length %d of argument 'sparse_indices'." % (len(sparse_shape), _attr_Nsparse))
    input_order = _execute.make_str(input_order, 'input_order')
    hashed_output = _execute.make_bool(hashed_output, 'hashed_output')
    num_buckets = _execute.make_int(num_buckets, 'num_buckets')
    hash_key = _execute.make_int(hash_key, 'hash_key')
    out_values_type = _execute.make_type(out_values_type, 'out_values_type')
    out_row_splits_type = _execute.make_type(out_row_splits_type, 'out_row_splits_type')
    _attr_ragged_values_types, ragged_values = _execute.convert_to_mixed_eager_tensors(ragged_values, ctx)
    _attr_ragged_splits_types, ragged_row_splits = _execute.convert_to_mixed_eager_tensors(ragged_row_splits, ctx)
    _attr_sparse_values_types, sparse_values = _execute.convert_to_mixed_eager_tensors(sparse_values, ctx)
    _attr_dense_types, dense_inputs = _execute.convert_to_mixed_eager_tensors(dense_inputs, ctx)
    sparse_indices = _ops.convert_n_to_tensor(sparse_indices, _dtypes.int64)
    sparse_shape = _ops.convert_n_to_tensor(sparse_shape, _dtypes.int64)
    _inputs_flat = list(ragged_values) + list(ragged_row_splits) + list(sparse_indices) + list(sparse_values) + list(sparse_shape) + list(dense_inputs)
    _attrs = ('Nsparse', _attr_Nsparse, 'input_order', input_order, 'hashed_output', hashed_output, 'num_buckets', num_buckets, 'hash_key', hash_key, 'ragged_values_types', _attr_ragged_values_types, 'ragged_splits_types', _attr_ragged_splits_types, 'sparse_values_types', _attr_sparse_values_types, 'dense_types', _attr_dense_types, 'out_values_type', out_values_type, 'out_row_splits_type', out_row_splits_type)
    _result = _execute.execute(b'RaggedCross', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RaggedCross', _inputs_flat, _attrs, _result)
    _result = _RaggedCrossOutput._make(_result)
    return _result