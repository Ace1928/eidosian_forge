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
def quantize_and_dequantize_v4_grad(gradients: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV4Grad_T], input: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV4Grad_T], input_min: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV4Grad_T], input_max: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV4Grad_T], axis: int=-1, name=None):
    """Returns the gradient of `QuantizeAndDequantizeV4`.

  Returns a gradient of 1 for inputs that are within the quantization range,
  or 0 otherwise.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    input: A `Tensor`. Must have the same type as `gradients`.
    input_min: A `Tensor`. Must have the same type as `gradients`.
    input_max: A `Tensor`. Must have the same type as `gradients`.
    axis: An optional `int`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (input_backprop, input_min_backprop, input_max_backprop).

    input_backprop: A `Tensor`. Has the same type as `gradients`.
    input_min_backprop: A `Tensor`. Has the same type as `gradients`.
    input_max_backprop: A `Tensor`. Has the same type as `gradients`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizeAndDequantizeV4Grad', name, gradients, input, input_min, input_max, 'axis', axis)
            _result = _QuantizeAndDequantizeV4GradOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantize_and_dequantize_v4_grad_eager_fallback(gradients, input, input_min, input_max, axis=axis, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if axis is None:
        axis = -1
    axis = _execute.make_int(axis, 'axis')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizeAndDequantizeV4Grad', gradients=gradients, input=input, input_min=input_min, input_max=input_max, axis=axis, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'axis', _op._get_attr_int('axis'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizeAndDequantizeV4Grad', _inputs_flat, _attrs, _result)
    _result = _QuantizeAndDequantizeV4GradOutput._make(_result)
    return _result