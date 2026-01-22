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
def sparse_concat_eager_fallback(indices: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], values: List[_atypes.TensorFuzzingAnnotation[TV_SparseConcat_T]], shapes: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], concat_dim: int, name, ctx):
    if not isinstance(indices, (list, tuple)):
        raise TypeError("Expected list for 'indices' argument to 'sparse_concat' Op, not %r." % indices)
    _attr_N = len(indices)
    if not isinstance(values, (list, tuple)):
        raise TypeError("Expected list for 'values' argument to 'sparse_concat' Op, not %r." % values)
    if len(values) != _attr_N:
        raise ValueError("List argument 'values' to 'sparse_concat' Op with length %d must match length %d of argument 'indices'." % (len(values), _attr_N))
    if not isinstance(shapes, (list, tuple)):
        raise TypeError("Expected list for 'shapes' argument to 'sparse_concat' Op, not %r." % shapes)
    if len(shapes) != _attr_N:
        raise ValueError("List argument 'shapes' to 'sparse_concat' Op with length %d must match length %d of argument 'indices'." % (len(shapes), _attr_N))
    concat_dim = _execute.make_int(concat_dim, 'concat_dim')
    _attr_T, values = _execute.args_to_matching_eager(list(values), ctx, [])
    indices = _ops.convert_n_to_tensor(indices, _dtypes.int64)
    shapes = _ops.convert_n_to_tensor(shapes, _dtypes.int64)
    _inputs_flat = list(indices) + list(values) + list(shapes)
    _attrs = ('concat_dim', concat_dim, 'N', _attr_N, 'T', _attr_T)
    _result = _execute.execute(b'SparseConcat', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseConcat', _inputs_flat, _attrs, _result)
    _result = _SparseConcatOutput._make(_result)
    return _result