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
def ragged_tensor_to_tensor(shape: _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToTensor_Tshape], values: _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToTensor_T], default_value: _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToTensor_T], row_partition_tensors: List[_atypes.TensorFuzzingAnnotation[TV_RaggedTensorToTensor_Tindex]], row_partition_types, name=None) -> _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToTensor_T]:
    """Create a dense tensor from a ragged tensor, possibly altering its shape.

  The `ragged_to_dense` op creates a dense tensor from a list of row partition
  tensors, a value vector, and default values. If the shape is unspecified, the
  minimal shape required to contain all the elements in the ragged tensor (the
  natural shape) will be used. If some dimensions are left unspecified, then the
  size of the natural shape is used in that dimension.

  The default_value will be broadcast to the output shape. After that, the values
  from the ragged tensor overwrite the default values. Note that the default_value
  must have less dimensions than the value.

  The row partition tensors are in the order of the dimensions.
  At present, the types can be:
  * "ROW_SPLITS": the row_splits tensor from the ragged tensor.
  * "VALUE_ROWIDS": the value_rowids tensor from the ragged tensor.
  * "FIRST_DIM_SIZE": if value_rowids is used for the first dimension, then it
    is preceded by "FIRST_DIM_SIZE".

  Args:
    shape: A `Tensor`. Must be one of the following types: `int64`, `int32`.
      The desired shape of the output tensor. If left unspecified (empty),
      the minimal shape required to contain all the elements in the ragged tensor
      (the natural shape) will be used. If some dimensions are left unspecified, then
      the size of the natural shape is used in that dimension.

      Note that dense dimensions cannot be modified by the shape argument. Trying to
      change the size of a dense dimension will cause the op to fail.
      Examples:
      natural shape: [4, 5, 6]
      shape: -1
      output shape: [4, 5, 6]

      natural shape: [4, 5, 6]
      shape: [3, -1, 2]
      output shape: [3, 5, 2]

      natural shape: [4, 5, 6]
      shape: [3, 7, 2]
      output shape: [3, 7, 2]
    values: A `Tensor`.
      A 1D tensor representing the values of the ragged tensor.
    default_value: A `Tensor`. Must have the same type as `values`.
      The default_value when the shape is larger than the ragged tensor. The
      default_value is broadcast until it is the shape of the output tensor, and
      then overwritten by values in the ragged tensor. The default value must be
      compatible with this broadcast operation, and must have fewer dimensions than
      the value tensor.
    row_partition_tensors: A list of at least 1 `Tensor` objects with the same type in: `int64`, `int32`.
    row_partition_types: A list of `strings`.
      The types of the row partition tensors. At present, these can be:
      * "ROW_SPLITS": the row_splits tensor from the ragged tensor.
      * "VALUE_ROWIDS": the value_rowids tensor from the ragged tensor.
      * "FIRST_DIM_SIZE": if value_rowids is used for the first dimension, then it
        is preceeded by "FIRST_DIM_SIZE".
      The tensors are in the order of the dimensions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `values`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RaggedTensorToTensor', name, shape, values, default_value, row_partition_tensors, 'row_partition_types', row_partition_types)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return ragged_tensor_to_tensor_eager_fallback(shape, values, default_value, row_partition_tensors, row_partition_types=row_partition_types, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(row_partition_tensors, (list, tuple)):
        raise TypeError("Expected list for 'row_partition_tensors' argument to 'ragged_tensor_to_tensor' Op, not %r." % row_partition_tensors)
    _attr_num_row_partition_tensors = len(row_partition_tensors)
    if not isinstance(row_partition_types, (list, tuple)):
        raise TypeError("Expected list for 'row_partition_types' argument to 'ragged_tensor_to_tensor' Op, not %r." % row_partition_types)
    row_partition_types = [_execute.make_str(_s, 'row_partition_types') for _s in row_partition_types]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RaggedTensorToTensor', shape=shape, values=values, default_value=default_value, row_partition_tensors=row_partition_tensors, row_partition_types=row_partition_types, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindex', _op._get_attr_type('Tindex'), 'Tshape', _op._get_attr_type('Tshape'), 'num_row_partition_tensors', _op._get_attr_int('num_row_partition_tensors'), 'row_partition_types', _op.get_attr('row_partition_types'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RaggedTensorToTensor', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result