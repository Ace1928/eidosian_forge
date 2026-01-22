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
def ragged_gather(params_nested_splits: List[_atypes.TensorFuzzingAnnotation[TV_RaggedGather_Tsplits]], params_dense_values: _atypes.TensorFuzzingAnnotation[TV_RaggedGather_Tvalues], indices: _atypes.TensorFuzzingAnnotation[TV_RaggedGather_Tindices], OUTPUT_RAGGED_RANK: int, name=None):
    """Gather ragged slices from `params` axis `0` according to `indices`.

  Outputs a `RaggedTensor` output composed from `output_dense_values` and
  `output_nested_splits`, such that:

  ```python
  output.shape = indices.shape + params.shape[1:]
  output.ragged_rank = indices.shape.ndims + params.ragged_rank
  output[i...j, d0...dn] = params[indices[i...j], d0...dn]
  ```

  where

  * `params =
     ragged.from_nested_row_splits(params_dense_values, params_nested_splits)`
     provides the values that should be gathered.
  * `indices` ia a dense tensor with dtype `int32` or `int64`, indicating which
     values should be gathered.
  * `output =
     ragged.from_nested_row_splits(output_dense_values, output_nested_splits)`
     is the output tensor.

  (Note: This c++ op is used to implement the higher-level python
  `tf.ragged.gather` op, which also supports ragged indices.)

  Args:
    params_nested_splits: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
      The `nested_row_splits` tensors that define the row-partitioning for the
      `params` RaggedTensor input.
    params_dense_values: A `Tensor`.
      The `flat_values` for the `params` RaggedTensor. There was a terminology change
      at the python level from dense_values to flat_values, so dense_values is the
      deprecated name.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Indices in the outermost dimension of `params` of the values that should be
      gathered.
    OUTPUT_RAGGED_RANK: An `int` that is `>= 0`.
      The ragged rank of the output RaggedTensor. `output_nested_splits` will contain
      this number of `row_splits` tensors. This value should equal
      `indices.shape.ndims + params.ragged_rank - 1`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_nested_splits, output_dense_values).

    output_nested_splits: A list of `OUTPUT_RAGGED_RANK` `Tensor` objects with the same type as `params_nested_splits`.
    output_dense_values: A `Tensor`. Has the same type as `params_dense_values`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RaggedGather', name, params_nested_splits, params_dense_values, indices, 'OUTPUT_RAGGED_RANK', OUTPUT_RAGGED_RANK)
            _result = _RaggedGatherOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return ragged_gather_eager_fallback(params_nested_splits, params_dense_values, indices, OUTPUT_RAGGED_RANK=OUTPUT_RAGGED_RANK, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(params_nested_splits, (list, tuple)):
        raise TypeError("Expected list for 'params_nested_splits' argument to 'ragged_gather' Op, not %r." % params_nested_splits)
    _attr_PARAMS_RAGGED_RANK = len(params_nested_splits)
    OUTPUT_RAGGED_RANK = _execute.make_int(OUTPUT_RAGGED_RANK, 'OUTPUT_RAGGED_RANK')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RaggedGather', params_nested_splits=params_nested_splits, params_dense_values=params_dense_values, indices=indices, OUTPUT_RAGGED_RANK=OUTPUT_RAGGED_RANK, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tvalues', _op._get_attr_type('Tvalues'), 'Tindices', _op._get_attr_type('Tindices'), 'Tsplits', _op._get_attr_type('Tsplits'), 'PARAMS_RAGGED_RANK', _op._get_attr_int('PARAMS_RAGGED_RANK'), 'OUTPUT_RAGGED_RANK', _op._get_attr_int('OUTPUT_RAGGED_RANK'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RaggedGather', _inputs_flat, _attrs, _result)
    _result = [_result[:OUTPUT_RAGGED_RANK]] + _result[OUTPUT_RAGGED_RANK:]
    _result = _RaggedGatherOutput._make(_result)
    return _result