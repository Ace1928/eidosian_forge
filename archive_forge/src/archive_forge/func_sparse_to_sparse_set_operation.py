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
def sparse_to_sparse_set_operation(set1_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], set1_values: _atypes.TensorFuzzingAnnotation[TV_SparseToSparseSetOperation_T], set1_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], set2_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], set2_values: _atypes.TensorFuzzingAnnotation[TV_SparseToSparseSetOperation_T], set2_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], set_operation: str, validate_indices: bool=True, name=None):
    """Applies set operation along last dimension of 2 `SparseTensor` inputs.

  See SetOperationOp::SetOperationFromContext for values of `set_operation`.

  If `validate_indices` is `True`, `SparseToSparseSetOperation` validates the
  order and range of `set1` and `set2` indices.

  Input `set1` is a `SparseTensor` represented by `set1_indices`, `set1_values`,
  and `set1_shape`. For `set1` ranked `n`, 1st `n-1` dimensions must be the same
  as `set2`. Dimension `n` contains values in a set, duplicates are allowed but
  ignored.

  Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
  and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
  as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
  ignored.

  If `validate_indices` is `True`, this op validates the order and range of `set1`
  and `set2` indices.

  Output `result` is a `SparseTensor` represented by `result_indices`,
  `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
  has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
  dimension contains the result of `set_operation` applied to the corresponding
  `[0...n-1]` dimension of `set`.

  Args:
    set1_indices: A `Tensor` of type `int64`.
      2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
      order.
    set1_values: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `string`.
      1D `Tensor`, values of a `SparseTensor`. Must be in row-major
      order.
    set1_shape: A `Tensor` of type `int64`.
      1D `Tensor`, shape of a `SparseTensor`. `set1_shape[0...n-1]` must
      be the same as `set2_shape[0...n-1]`, `set1_shape[n]` is the
      max set size across `0...n-1` dimensions.
    set2_indices: A `Tensor` of type `int64`.
      2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
      order.
    set2_values: A `Tensor`. Must have the same type as `set1_values`.
      1D `Tensor`, values of a `SparseTensor`. Must be in row-major
      order.
    set2_shape: A `Tensor` of type `int64`.
      1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
      be the same as `set1_shape[0...n-1]`, `set2_shape[n]` is the
      max set size across `0...n-1` dimensions.
    set_operation: A `string`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result_indices, result_values, result_shape).

    result_indices: A `Tensor` of type `int64`.
    result_values: A `Tensor`. Has the same type as `set1_values`.
    result_shape: A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseToSparseSetOperation', name, set1_indices, set1_values, set1_shape, set2_indices, set2_values, set2_shape, 'set_operation', set_operation, 'validate_indices', validate_indices)
            _result = _SparseToSparseSetOperationOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_to_sparse_set_operation_eager_fallback(set1_indices, set1_values, set1_shape, set2_indices, set2_values, set2_shape, set_operation=set_operation, validate_indices=validate_indices, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    set_operation = _execute.make_str(set_operation, 'set_operation')
    if validate_indices is None:
        validate_indices = True
    validate_indices = _execute.make_bool(validate_indices, 'validate_indices')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseToSparseSetOperation', set1_indices=set1_indices, set1_values=set1_values, set1_shape=set1_shape, set2_indices=set2_indices, set2_values=set2_values, set2_shape=set2_shape, set_operation=set_operation, validate_indices=validate_indices, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('set_operation', _op.get_attr('set_operation'), 'validate_indices', _op._get_attr_bool('validate_indices'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseToSparseSetOperation', _inputs_flat, _attrs, _result)
    _result = _SparseToSparseSetOperationOutput._make(_result)
    return _result