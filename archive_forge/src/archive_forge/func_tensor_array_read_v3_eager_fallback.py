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
def tensor_array_read_v3_eager_fallback(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], index: _atypes.TensorFuzzingAnnotation[_atypes.Int32], flow_in: _atypes.TensorFuzzingAnnotation[_atypes.Float32], dtype: TV_TensorArrayReadV3_dtype, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_TensorArrayReadV3_dtype]:
    dtype = _execute.make_type(dtype, 'dtype')
    handle = _ops.convert_to_tensor(handle, _dtypes.resource)
    index = _ops.convert_to_tensor(index, _dtypes.int32)
    flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
    _inputs_flat = [handle, index, flow_in]
    _attrs = ('dtype', dtype)
    _result = _execute.execute(b'TensorArrayReadV3', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TensorArrayReadV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result