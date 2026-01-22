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
def resource_gather_eager_fallback(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], indices: _atypes.TensorFuzzingAnnotation[TV_ResourceGather_Tindices], dtype: TV_ResourceGather_dtype, batch_dims: int, validate_indices: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_ResourceGather_dtype]:
    dtype = _execute.make_type(dtype, 'dtype')
    if batch_dims is None:
        batch_dims = 0
    batch_dims = _execute.make_int(batch_dims, 'batch_dims')
    if validate_indices is None:
        validate_indices = True
    validate_indices = _execute.make_bool(validate_indices, 'validate_indices')
    _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64])
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource, indices]
    _attrs = ('batch_dims', batch_dims, 'validate_indices', validate_indices, 'dtype', dtype, 'Tindices', _attr_Tindices)
    _result = _execute.execute(b'ResourceGather', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ResourceGather', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result