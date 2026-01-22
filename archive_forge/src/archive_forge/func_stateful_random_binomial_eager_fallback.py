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
def stateful_random_binomial_eager_fallback(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], algorithm: _atypes.TensorFuzzingAnnotation[_atypes.Int64], shape: _atypes.TensorFuzzingAnnotation[TV_StatefulRandomBinomial_S], counts: _atypes.TensorFuzzingAnnotation[TV_StatefulRandomBinomial_T], probs: _atypes.TensorFuzzingAnnotation[TV_StatefulRandomBinomial_T], dtype: TV_StatefulRandomBinomial_dtype, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_StatefulRandomBinomial_dtype]:
    if dtype is None:
        dtype = _dtypes.int64
    dtype = _execute.make_type(dtype, 'dtype')
    _attr_S, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64])
    _attr_T, _inputs_T = _execute.args_to_matching_eager([counts, probs], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64], _dtypes.float64)
    counts, probs = _inputs_T
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    algorithm = _ops.convert_to_tensor(algorithm, _dtypes.int64)
    _inputs_flat = [resource, algorithm, shape, counts, probs]
    _attrs = ('S', _attr_S, 'T', _attr_T, 'dtype', dtype)
    _result = _execute.execute(b'StatefulRandomBinomial', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('StatefulRandomBinomial', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result