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
def resource_scatter_nd_max_eager_fallback(ref: _atypes.TensorFuzzingAnnotation[_atypes.Resource], indices: _atypes.TensorFuzzingAnnotation[TV_ResourceScatterNdMax_Tindices], updates: _atypes.TensorFuzzingAnnotation[TV_ResourceScatterNdMax_T], use_locking: bool, name, ctx):
    if use_locking is None:
        use_locking = True
    use_locking = _execute.make_bool(use_locking, 'use_locking')
    _attr_T, (updates,) = _execute.args_to_matching_eager([updates], ctx, [])
    _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64])
    ref = _ops.convert_to_tensor(ref, _dtypes.resource)
    _inputs_flat = [ref, indices, updates]
    _attrs = ('T', _attr_T, 'Tindices', _attr_Tindices, 'use_locking', use_locking)
    _result = _execute.execute(b'ResourceScatterNdMax', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result