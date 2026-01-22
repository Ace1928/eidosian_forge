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
def sparse_cross_v2_eager_fallback(indices: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], values, shapes: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], dense_inputs, sep: _atypes.TensorFuzzingAnnotation[_atypes.String], name, ctx):
    if not isinstance(indices, (list, tuple)):
        raise TypeError("Expected list for 'indices' argument to 'sparse_cross_v2' Op, not %r." % indices)
    _attr_N = len(indices)
    if not isinstance(shapes, (list, tuple)):
        raise TypeError("Expected list for 'shapes' argument to 'sparse_cross_v2' Op, not %r." % shapes)
    if len(shapes) != _attr_N:
        raise ValueError("List argument 'shapes' to 'sparse_cross_v2' Op with length %d must match length %d of argument 'indices'." % (len(shapes), _attr_N))
    _attr_sparse_types, values = _execute.convert_to_mixed_eager_tensors(values, ctx)
    _attr_dense_types, dense_inputs = _execute.convert_to_mixed_eager_tensors(dense_inputs, ctx)
    indices = _ops.convert_n_to_tensor(indices, _dtypes.int64)
    shapes = _ops.convert_n_to_tensor(shapes, _dtypes.int64)
    sep = _ops.convert_to_tensor(sep, _dtypes.string)
    _inputs_flat = list(indices) + list(values) + list(shapes) + list(dense_inputs) + [sep]
    _attrs = ('N', _attr_N, 'sparse_types', _attr_sparse_types, 'dense_types', _attr_dense_types)
    _result = _execute.execute(b'SparseCrossV2', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseCrossV2', _inputs_flat, _attrs, _result)
    _result = _SparseCrossV2Output._make(_result)
    return _result