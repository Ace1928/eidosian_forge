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
def quantize_and_dequantize_v4(input: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV4_T], input_min: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV4_T], input_max: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV4_T], signed_input: bool=True, num_bits: int=8, range_given: bool=False, round_mode: str='HALF_TO_EVEN', narrow_range: bool=False, axis: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV4_T]:
    """Quantizes then dequantizes a tensor.

  This is almost identical to QuantizeAndDequantizeV2, except that it returns a
  gradient of 1 for inputs that are within the quantization range, or 0 otherwise.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      Tensor to quantize and then dequantize.
    input_min: A `Tensor`. Must have the same type as `input`.
      If `range_given == True`, this specifies the minimum input value that needs to
      be represented, otherwise it is determined from the min value of the `input`
      tensor.
    input_max: A `Tensor`. Must have the same type as `input`.
      If `range_given == True`, this specifies the maximum input value that needs to
      be represented, otherwise it is determined from the max value of the `input`
      tensor.
    signed_input: An optional `bool`. Defaults to `True`.
      Whether the quantization is signed or unsigned. (actually this parameter should
      have been called <b>`signed_output`</b>)
    num_bits: An optional `int`. Defaults to `8`.
      The bitwidth of the quantization.
    range_given: An optional `bool`. Defaults to `False`.
      Whether the range is given or should be determined from the `input` tensor.
    round_mode: An optional `string` from: `"HALF_TO_EVEN", "HALF_UP"`. Defaults to `"HALF_TO_EVEN"`.
      The 'round_mode' attribute controls which rounding tie-breaking algorithm is
      used when rounding float values to their quantized equivalents. The following
      rounding modes are currently supported:

      *   HALF_TO_EVEN: this is the default round_mode.
      *   HALF_UP: round towards positive. In this mode 7.5 rounds up to 8 and -7.5
          rounds up to -7.
    narrow_range: An optional `bool`. Defaults to `False`.
      If True, then the absolute value of the quantized minimum value is the same as
      the quantized maximum value, instead of 1 greater.
      i.e. for 8 bit quantization, the minimum value is -127 instead of -128.
    axis: An optional `int`. Defaults to `-1`.
      If specified, this axis is treated as a channel or slice axis, and a separate
      quantization range is used for each channel or slice along this axis.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizeAndDequantizeV4', name, input, input_min, input_max, 'signed_input', signed_input, 'num_bits', num_bits, 'range_given', range_given, 'round_mode', round_mode, 'narrow_range', narrow_range, 'axis', axis)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantize_and_dequantize_v4_eager_fallback(input, input_min, input_max, signed_input=signed_input, num_bits=num_bits, range_given=range_given, round_mode=round_mode, narrow_range=narrow_range, axis=axis, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if signed_input is None:
        signed_input = True
    signed_input = _execute.make_bool(signed_input, 'signed_input')
    if num_bits is None:
        num_bits = 8
    num_bits = _execute.make_int(num_bits, 'num_bits')
    if range_given is None:
        range_given = False
    range_given = _execute.make_bool(range_given, 'range_given')
    if round_mode is None:
        round_mode = 'HALF_TO_EVEN'
    round_mode = _execute.make_str(round_mode, 'round_mode')
    if narrow_range is None:
        narrow_range = False
    narrow_range = _execute.make_bool(narrow_range, 'narrow_range')
    if axis is None:
        axis = -1
    axis = _execute.make_int(axis, 'axis')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizeAndDequantizeV4', input=input, input_min=input_min, input_max=input_max, signed_input=signed_input, num_bits=num_bits, range_given=range_given, round_mode=round_mode, narrow_range=narrow_range, axis=axis, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('signed_input', _op._get_attr_bool('signed_input'), 'num_bits', _op._get_attr_int('num_bits'), 'range_given', _op._get_attr_bool('range_given'), 'T', _op._get_attr_type('T'), 'round_mode', _op.get_attr('round_mode'), 'narrow_range', _op._get_attr_bool('narrow_range'), 'axis', _op._get_attr_int('axis'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizeAndDequantizeV4', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result