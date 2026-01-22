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
def fractional_max_pool_grad(orig_input: _atypes.TensorFuzzingAnnotation[TV_FractionalMaxPoolGrad_T], orig_output: _atypes.TensorFuzzingAnnotation[TV_FractionalMaxPoolGrad_T], out_backprop: _atypes.TensorFuzzingAnnotation[TV_FractionalMaxPoolGrad_T], row_pooling_sequence: _atypes.TensorFuzzingAnnotation[_atypes.Int64], col_pooling_sequence: _atypes.TensorFuzzingAnnotation[_atypes.Int64], overlapping: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_FractionalMaxPoolGrad_T]:
    """Computes gradient of the FractionalMaxPool function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      Original input for `fractional_max_pool`
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      Original output for `fractional_max_pool`
    out_backprop: A `Tensor`. Must have the same type as `orig_input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients
      w.r.t. the output of `fractional_max_pool`.
    row_pooling_sequence: A `Tensor` of type `int64`.
      row pooling sequence, form pooling region with
      col_pooling_sequence.
    col_pooling_sequence: A `Tensor` of type `int64`.
      column pooling sequence, form pooling region with
      row_pooling sequence.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [20, 16] for fractional max pooling.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'FractionalMaxPoolGrad', name, orig_input, orig_output, out_backprop, row_pooling_sequence, col_pooling_sequence, 'overlapping', overlapping)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return fractional_max_pool_grad_eager_fallback(orig_input, orig_output, out_backprop, row_pooling_sequence, col_pooling_sequence, overlapping=overlapping, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if overlapping is None:
        overlapping = False
    overlapping = _execute.make_bool(overlapping, 'overlapping')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('FractionalMaxPoolGrad', orig_input=orig_input, orig_output=orig_output, out_backprop=out_backprop, row_pooling_sequence=row_pooling_sequence, col_pooling_sequence=col_pooling_sequence, overlapping=overlapping, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('overlapping', _op._get_attr_bool('overlapping'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('FractionalMaxPoolGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result