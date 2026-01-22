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
def non_deterministic_ints_eager_fallback(shape: _atypes.TensorFuzzingAnnotation[TV_NonDeterministicInts_shape_dtype], dtype: TV_NonDeterministicInts_dtype, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_NonDeterministicInts_dtype]:
    if dtype is None:
        dtype = _dtypes.int64
    dtype = _execute.make_type(dtype, 'dtype')
    _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], ctx, [], _dtypes.int64)
    _inputs_flat = [shape]
    _attrs = ('dtype', dtype, 'shape_dtype', _attr_shape_dtype)
    _result = _execute.execute(b'NonDeterministicInts', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('NonDeterministicInts', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result