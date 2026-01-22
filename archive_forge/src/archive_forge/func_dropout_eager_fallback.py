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
def dropout_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_Dropout_T], rate: _atypes.TensorFuzzingAnnotation[TV_Dropout_T], noise_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int32], seed1: _atypes.TensorFuzzingAnnotation[TV_Dropout_Tseed], seed2: _atypes.TensorFuzzingAnnotation[TV_Dropout_Tseed], name, ctx):
    _attr_T, _inputs_T = _execute.args_to_matching_eager([input, rate], ctx, [_dtypes.float32, _dtypes.half, _dtypes.float64])
    input, rate = _inputs_T
    _attr_Tseed, _inputs_Tseed = _execute.args_to_matching_eager([seed1, seed2], ctx, [_dtypes.int32, _dtypes.int64])
    seed1, seed2 = _inputs_Tseed
    noise_shape = _ops.convert_to_tensor(noise_shape, _dtypes.int32)
    _inputs_flat = [input, rate, noise_shape, seed1, seed2]
    _attrs = ('T', _attr_T, 'Tseed', _attr_Tseed)
    _result = _execute.execute(b'Dropout', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Dropout', _inputs_flat, _attrs, _result)
    _result = _DropoutOutput._make(_result)
    return _result