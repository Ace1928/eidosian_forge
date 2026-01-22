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
def quantize_and_dequantize_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantize_T], signed_input: bool, num_bits: int, range_given: bool, input_min: float, input_max: float, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantize_T]:
    if signed_input is None:
        signed_input = True
    signed_input = _execute.make_bool(signed_input, 'signed_input')
    if num_bits is None:
        num_bits = 8
    num_bits = _execute.make_int(num_bits, 'num_bits')
    if range_given is None:
        range_given = False
    range_given = _execute.make_bool(range_given, 'range_given')
    if input_min is None:
        input_min = 0
    input_min = _execute.make_float(input_min, 'input_min')
    if input_max is None:
        input_max = 0
    input_max = _execute.make_float(input_max, 'input_max')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64])
    _inputs_flat = [input]
    _attrs = ('signed_input', signed_input, 'num_bits', num_bits, 'range_given', range_given, 'input_min', input_min, 'input_max', input_max, 'T', _attr_T)
    _result = _execute.execute(b'QuantizeAndDequantize', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizeAndDequantize', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result