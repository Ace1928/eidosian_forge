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
def lin_space_eager_fallback(start: _atypes.TensorFuzzingAnnotation[TV_LinSpace_T], stop: _atypes.TensorFuzzingAnnotation[TV_LinSpace_T], num: _atypes.TensorFuzzingAnnotation[TV_LinSpace_Tidx], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_LinSpace_T]:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([start, stop], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64])
    start, stop = _inputs_T
    _attr_Tidx, (num,) = _execute.args_to_matching_eager([num], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _inputs_flat = [start, stop, num]
    _attrs = ('T', _attr_T, 'Tidx', _attr_Tidx)
    _result = _execute.execute(b'LinSpace', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('LinSpace', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result