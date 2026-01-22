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
def collective_reduce_v3_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_CollectiveReduceV3_T], communicator: _atypes.TensorFuzzingAnnotation[_atypes.Resource], group_assignment: _atypes.TensorFuzzingAnnotation[_atypes.Int32], reduction: str, timeout_seconds: float, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_CollectiveReduceV3_T]:
    reduction = _execute.make_str(reduction, 'reduction')
    if timeout_seconds is None:
        timeout_seconds = 0
    timeout_seconds = _execute.make_float(timeout_seconds, 'timeout_seconds')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bfloat16, _dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64])
    communicator = _ops.convert_to_tensor(communicator, _dtypes.resource)
    group_assignment = _ops.convert_to_tensor(group_assignment, _dtypes.int32)
    _inputs_flat = [input, communicator, group_assignment]
    _attrs = ('T', _attr_T, 'reduction', reduction, 'timeout_seconds', timeout_seconds)
    _result = _execute.execute(b'CollectiveReduceV3', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CollectiveReduceV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result