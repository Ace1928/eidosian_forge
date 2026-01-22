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
def strided_slice_assign(ref: _atypes.TensorFuzzingAnnotation[TV_StridedSliceAssign_T], begin: _atypes.TensorFuzzingAnnotation[TV_StridedSliceAssign_Index], end: _atypes.TensorFuzzingAnnotation[TV_StridedSliceAssign_Index], strides: _atypes.TensorFuzzingAnnotation[TV_StridedSliceAssign_Index], value: _atypes.TensorFuzzingAnnotation[TV_StridedSliceAssign_T], begin_mask: int=0, end_mask: int=0, ellipsis_mask: int=0, new_axis_mask: int=0, shrink_axis_mask: int=0, name=None) -> _atypes.TensorFuzzingAnnotation[TV_StridedSliceAssign_T]:
    """Assign `value` to the sliced l-value reference of `ref`.

  The values of `value` are assigned to the positions in the variable
  `ref` that are selected by the slice parameters. The slice parameters
  `begin`, `end`, `strides`, etc. work exactly as in `StridedSlice`.

  NOTE this op currently does not support broadcasting and so `value`'s
  shape must be exactly the shape produced by the slice of `ref`.

  Args:
    ref: A mutable `Tensor`.
    begin: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    end: A `Tensor`. Must have the same type as `begin`.
    strides: A `Tensor`. Must have the same type as `begin`.
    value: A `Tensor`. Must have the same type as `ref`.
    begin_mask: An optional `int`. Defaults to `0`.
    end_mask: An optional `int`. Defaults to `0`.
    ellipsis_mask: An optional `int`. Defaults to `0`.
    new_axis_mask: An optional `int`. Defaults to `0`.
    shrink_axis_mask: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `ref`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("strided_slice_assign op does not support eager execution. Arg 'output_ref' is a ref.")
    if begin_mask is None:
        begin_mask = 0
    begin_mask = _execute.make_int(begin_mask, 'begin_mask')
    if end_mask is None:
        end_mask = 0
    end_mask = _execute.make_int(end_mask, 'end_mask')
    if ellipsis_mask is None:
        ellipsis_mask = 0
    ellipsis_mask = _execute.make_int(ellipsis_mask, 'ellipsis_mask')
    if new_axis_mask is None:
        new_axis_mask = 0
    new_axis_mask = _execute.make_int(new_axis_mask, 'new_axis_mask')
    if shrink_axis_mask is None:
        shrink_axis_mask = 0
    shrink_axis_mask = _execute.make_int(shrink_axis_mask, 'shrink_axis_mask')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StridedSliceAssign', ref=ref, begin=begin, end=end, strides=strides, value=value, begin_mask=begin_mask, end_mask=end_mask, ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Index', _op._get_attr_type('Index'), 'begin_mask', _op._get_attr_int('begin_mask'), 'end_mask', _op._get_attr_int('end_mask'), 'ellipsis_mask', _op._get_attr_int('ellipsis_mask'), 'new_axis_mask', _op._get_attr_int('new_axis_mask'), 'shrink_axis_mask', _op._get_attr_int('shrink_axis_mask'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StridedSliceAssign', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result