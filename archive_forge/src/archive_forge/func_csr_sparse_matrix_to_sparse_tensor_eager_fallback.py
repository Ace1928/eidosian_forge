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
def csr_sparse_matrix_to_sparse_tensor_eager_fallback(sparse_matrix: _atypes.TensorFuzzingAnnotation[_atypes.Variant], type: TV_CSRSparseMatrixToSparseTensor_type, name, ctx):
    type = _execute.make_type(type, 'type')
    sparse_matrix = _ops.convert_to_tensor(sparse_matrix, _dtypes.variant)
    _inputs_flat = [sparse_matrix]
    _attrs = ('type', type)
    _result = _execute.execute(b'CSRSparseMatrixToSparseTensor', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CSRSparseMatrixToSparseTensor', _inputs_flat, _attrs, _result)
    _result = _CSRSparseMatrixToSparseTensorOutput._make(_result)
    return _result