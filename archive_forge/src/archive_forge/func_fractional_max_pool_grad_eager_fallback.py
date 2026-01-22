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
def fractional_max_pool_grad_eager_fallback(orig_input: _atypes.TensorFuzzingAnnotation[TV_FractionalMaxPoolGrad_T], orig_output: _atypes.TensorFuzzingAnnotation[TV_FractionalMaxPoolGrad_T], out_backprop: _atypes.TensorFuzzingAnnotation[TV_FractionalMaxPoolGrad_T], row_pooling_sequence: _atypes.TensorFuzzingAnnotation[_atypes.Int64], col_pooling_sequence: _atypes.TensorFuzzingAnnotation[_atypes.Int64], overlapping: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_FractionalMaxPoolGrad_T]:
    if overlapping is None:
        overlapping = False
    overlapping = _execute.make_bool(overlapping, 'overlapping')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([orig_input, orig_output, out_backprop], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64])
    orig_input, orig_output, out_backprop = _inputs_T
    row_pooling_sequence = _ops.convert_to_tensor(row_pooling_sequence, _dtypes.int64)
    col_pooling_sequence = _ops.convert_to_tensor(col_pooling_sequence, _dtypes.int64)
    _inputs_flat = [orig_input, orig_output, out_backprop, row_pooling_sequence, col_pooling_sequence]
    _attrs = ('overlapping', overlapping, 'T', _attr_T)
    _result = _execute.execute(b'FractionalMaxPoolGrad', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('FractionalMaxPoolGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result