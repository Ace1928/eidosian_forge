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
def lu_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_Lu_T], output_idx_type: TV_Lu_output_idx_type, name, ctx):
    if output_idx_type is None:
        output_idx_type = _dtypes.int32
    output_idx_type = _execute.make_type(output_idx_type, 'output_idx_type')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float64, _dtypes.float32, _dtypes.half, _dtypes.complex64, _dtypes.complex128])
    _inputs_flat = [input]
    _attrs = ('T', _attr_T, 'output_idx_type', output_idx_type)
    _result = _execute.execute(b'Lu', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Lu', _inputs_flat, _attrs, _result)
    _result = _LuOutput._make(_result)
    return _result