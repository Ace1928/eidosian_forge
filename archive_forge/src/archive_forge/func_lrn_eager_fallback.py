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
def lrn_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_LRN_T], depth_radius: int, bias: float, alpha: float, beta: float, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_LRN_T]:
    if depth_radius is None:
        depth_radius = 5
    depth_radius = _execute.make_int(depth_radius, 'depth_radius')
    if bias is None:
        bias = 1
    bias = _execute.make_float(bias, 'bias')
    if alpha is None:
        alpha = 1
    alpha = _execute.make_float(alpha, 'alpha')
    if beta is None:
        beta = 0.5
    beta = _execute.make_float(beta, 'beta')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32], _dtypes.float32)
    _inputs_flat = [input]
    _attrs = ('depth_radius', depth_radius, 'bias', bias, 'alpha', alpha, 'beta', beta, 'T', _attr_T)
    _result = _execute.execute(b'LRN', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('LRN', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result