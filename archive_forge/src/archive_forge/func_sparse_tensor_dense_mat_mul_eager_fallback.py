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
def sparse_tensor_dense_mat_mul_eager_fallback(a_indices: _atypes.TensorFuzzingAnnotation[TV_SparseTensorDenseMatMul_Tindices], a_values: _atypes.TensorFuzzingAnnotation[TV_SparseTensorDenseMatMul_T], a_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], b: _atypes.TensorFuzzingAnnotation[TV_SparseTensorDenseMatMul_T], adjoint_a: bool, adjoint_b: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_SparseTensorDenseMatMul_T]:
    if adjoint_a is None:
        adjoint_a = False
    adjoint_a = _execute.make_bool(adjoint_a, 'adjoint_a')
    if adjoint_b is None:
        adjoint_b = False
    adjoint_b = _execute.make_bool(adjoint_b, 'adjoint_b')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([a_values, b], ctx, [])
    a_values, b = _inputs_T
    _attr_Tindices, (a_indices,) = _execute.args_to_matching_eager([a_indices], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int64)
    a_shape = _ops.convert_to_tensor(a_shape, _dtypes.int64)
    _inputs_flat = [a_indices, a_values, a_shape, b]
    _attrs = ('T', _attr_T, 'Tindices', _attr_Tindices, 'adjoint_a', adjoint_a, 'adjoint_b', adjoint_b)
    _result = _execute.execute(b'SparseTensorDenseMatMul', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseTensorDenseMatMul', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result