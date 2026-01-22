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
def ragged_count_sparse_output_eager_fallback(splits: _atypes.TensorFuzzingAnnotation[_atypes.Int64], values: _atypes.TensorFuzzingAnnotation[TV_RaggedCountSparseOutput_T], weights: _atypes.TensorFuzzingAnnotation[TV_RaggedCountSparseOutput_output_type], binary_output: bool, minlength: int, maxlength: int, name, ctx):
    binary_output = _execute.make_bool(binary_output, 'binary_output')
    if minlength is None:
        minlength = -1
    minlength = _execute.make_int(minlength, 'minlength')
    if maxlength is None:
        maxlength = -1
    maxlength = _execute.make_int(maxlength, 'maxlength')
    _attr_T, (values,) = _execute.args_to_matching_eager([values], ctx, [_dtypes.int32, _dtypes.int64])
    _attr_output_type, (weights,) = _execute.args_to_matching_eager([weights], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64])
    splits = _ops.convert_to_tensor(splits, _dtypes.int64)
    _inputs_flat = [splits, values, weights]
    _attrs = ('T', _attr_T, 'minlength', minlength, 'maxlength', maxlength, 'binary_output', binary_output, 'output_type', _attr_output_type)
    _result = _execute.execute(b'RaggedCountSparseOutput', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RaggedCountSparseOutput', _inputs_flat, _attrs, _result)
    _result = _RaggedCountSparseOutputOutput._make(_result)
    return _result