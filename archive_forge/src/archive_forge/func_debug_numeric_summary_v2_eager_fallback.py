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
def debug_numeric_summary_v2_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_DebugNumericSummaryV2_T], output_dtype: TV_DebugNumericSummaryV2_output_dtype, tensor_debug_mode: int, tensor_id: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_DebugNumericSummaryV2_output_dtype]:
    if output_dtype is None:
        output_dtype = _dtypes.float32
    output_dtype = _execute.make_type(output_dtype, 'output_dtype')
    if tensor_debug_mode is None:
        tensor_debug_mode = -1
    tensor_debug_mode = _execute.make_int(tensor_debug_mode, 'tensor_debug_mode')
    if tensor_id is None:
        tensor_id = -1
    tensor_id = _execute.make_int(tensor_id, 'tensor_id')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
    _inputs_flat = [input]
    _attrs = ('output_dtype', output_dtype, 'T', _attr_T, 'tensor_debug_mode', tensor_debug_mode, 'tensor_id', tensor_id)
    _result = _execute.execute(b'DebugNumericSummaryV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DebugNumericSummaryV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result