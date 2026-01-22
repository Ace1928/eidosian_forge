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
def split_v(value: _atypes.TensorFuzzingAnnotation[TV_SplitV_T], size_splits: _atypes.TensorFuzzingAnnotation[TV_SplitV_Tlen], axis: _atypes.TensorFuzzingAnnotation[_atypes.Int32], num_split: int, name=None):
    """Splits a tensor into `num_split` tensors along one dimension.

  Args:
    value: A `Tensor`. The tensor to split.
    size_splits: A `Tensor`. Must be one of the following types: `int8`, `int32`, `int64`.
      list containing the sizes of each output tensor along the split
      dimension. Must sum to the dimension of value along split_dim.
      Can contain one -1 indicating that dimension is to be inferred.
    axis: A `Tensor` of type `int32`.
      0-D.  The dimension along which to split.  Must be in the range
      `[-rank(value), rank(value))`.
    num_split: An `int` that is `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `num_split` `Tensor` objects with the same type as `value`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SplitV', name, value, size_splits, axis, 'num_split', num_split)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return split_v_eager_fallback(value, size_splits, axis, num_split=num_split, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    num_split = _execute.make_int(num_split, 'num_split')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SplitV', value=value, size_splits=size_splits, split_dim=axis, num_split=num_split, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('num_split', _op._get_attr_int('num_split'), 'T', _op._get_attr_type('T'), 'Tlen', _op._get_attr_type('Tlen'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SplitV', _inputs_flat, _attrs, _result)
    return _result