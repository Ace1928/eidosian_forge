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
def sparse_reshape_eager_fallback(input_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], input_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], new_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name, ctx):
    input_indices = _ops.convert_to_tensor(input_indices, _dtypes.int64)
    input_shape = _ops.convert_to_tensor(input_shape, _dtypes.int64)
    new_shape = _ops.convert_to_tensor(new_shape, _dtypes.int64)
    _inputs_flat = [input_indices, input_shape, new_shape]
    _attrs = None
    _result = _execute.execute(b'SparseReshape', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseReshape', _inputs_flat, _attrs, _result)
    _result = _SparseReshapeOutput._make(_result)
    return _result