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
def stateless_random_uniform_full_int_eager_fallback(shape: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomUniformFullInt_T], seed: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomUniformFullInt_Tseed], dtype: TV_StatelessRandomUniformFullInt_dtype, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_StatelessRandomUniformFullInt_dtype]:
    if dtype is None:
        dtype = _dtypes.uint64
    dtype = _execute.make_type(dtype, 'dtype')
    _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.uint32, _dtypes.uint64], _dtypes.int64)
    _inputs_flat = [shape, seed]
    _attrs = ('dtype', dtype, 'T', _attr_T, 'Tseed', _attr_Tseed)
    _result = _execute.execute(b'StatelessRandomUniformFullInt', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('StatelessRandomUniformFullInt', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result