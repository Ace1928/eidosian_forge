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
def batch_mat_mul_v2_eager_fallback(x: _atypes.TensorFuzzingAnnotation[TV_BatchMatMulV2_T], y: _atypes.TensorFuzzingAnnotation[TV_BatchMatMulV2_T], adj_x: bool, adj_y: bool, grad_x: bool, grad_y: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_BatchMatMulV2_T]:
    if adj_x is None:
        adj_x = False
    adj_x = _execute.make_bool(adj_x, 'adj_x')
    if adj_y is None:
        adj_y = False
    adj_y = _execute.make_bool(adj_y, 'adj_y')
    if grad_x is None:
        grad_x = False
    grad_x = _execute.make_bool(grad_x, 'grad_x')
    if grad_y is None:
        grad_y = False
    grad_y = _execute.make_bool(grad_y, 'grad_y')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, _dtypes.complex64, _dtypes.complex128])
    x, y = _inputs_T
    _inputs_flat = [x, y]
    _attrs = ('T', _attr_T, 'adj_x', adj_x, 'adj_y', adj_y, 'grad_x', grad_x, 'grad_y', grad_y)
    _result = _execute.execute(b'BatchMatMulV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BatchMatMulV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result