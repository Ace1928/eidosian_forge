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
def batch_cholesky_grad_eager_fallback(l: _atypes.TensorFuzzingAnnotation[TV_BatchCholeskyGrad_T], grad: _atypes.TensorFuzzingAnnotation[TV_BatchCholeskyGrad_T], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_BatchCholeskyGrad_T]:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([l, grad], ctx, [_dtypes.float32, _dtypes.float64])
    l, grad = _inputs_T
    _inputs_flat = [l, grad]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'BatchCholeskyGrad', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BatchCholeskyGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result