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
def sparse_to_dense_eager_fallback(sparse_indices: _atypes.TensorFuzzingAnnotation[TV_SparseToDense_Tindices], output_shape: _atypes.TensorFuzzingAnnotation[TV_SparseToDense_Tindices], sparse_values: _atypes.TensorFuzzingAnnotation[TV_SparseToDense_T], default_value: _atypes.TensorFuzzingAnnotation[TV_SparseToDense_T], validate_indices: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_SparseToDense_T]:
    if validate_indices is None:
        validate_indices = True
    validate_indices = _execute.make_bool(validate_indices, 'validate_indices')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([sparse_values, default_value], ctx, [])
    sparse_values, default_value = _inputs_T
    _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([sparse_indices, output_shape], ctx, [_dtypes.int32, _dtypes.int64])
    sparse_indices, output_shape = _inputs_Tindices
    _inputs_flat = [sparse_indices, output_shape, sparse_values, default_value]
    _attrs = ('validate_indices', validate_indices, 'T', _attr_T, 'Tindices', _attr_Tindices)
    _result = _execute.execute(b'SparseToDense', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseToDense', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result