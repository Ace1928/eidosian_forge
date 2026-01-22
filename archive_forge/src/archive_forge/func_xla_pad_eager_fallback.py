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
def xla_pad_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_XlaPad_T], padding_value: _atypes.TensorFuzzingAnnotation[TV_XlaPad_T], padding_low: _atypes.TensorFuzzingAnnotation[TV_XlaPad_Tindices], padding_high: _atypes.TensorFuzzingAnnotation[TV_XlaPad_Tindices], padding_interior: _atypes.TensorFuzzingAnnotation[TV_XlaPad_Tindices], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_XlaPad_T]:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([input, padding_value], ctx, [])
    input, padding_value = _inputs_T
    _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([padding_low, padding_high, padding_interior], ctx, [_dtypes.int32, _dtypes.int64])
    padding_low, padding_high, padding_interior = _inputs_Tindices
    _inputs_flat = [input, padding_value, padding_low, padding_high, padding_interior]
    _attrs = ('T', _attr_T, 'Tindices', _attr_Tindices)
    _result = _execute.execute(b'XlaPad', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaPad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result