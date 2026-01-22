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
def max_pool_grad_with_argmax_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_MaxPoolGradWithArgmax_T], grad: _atypes.TensorFuzzingAnnotation[TV_MaxPoolGradWithArgmax_T], argmax: _atypes.TensorFuzzingAnnotation[TV_MaxPoolGradWithArgmax_Targmax], ksize, strides, padding: str, include_batch_in_index: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_MaxPoolGradWithArgmax_T]:
    if not isinstance(ksize, (list, tuple)):
        raise TypeError("Expected list for 'ksize' argument to 'max_pool_grad_with_argmax' Op, not %r." % ksize)
    ksize = [_execute.make_int(_i, 'ksize') for _i in ksize]
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'max_pool_grad_with_argmax' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if include_batch_in_index is None:
        include_batch_in_index = False
    include_batch_in_index = _execute.make_bool(include_batch_in_index, 'include_batch_in_index')
    _attr_Targmax, (argmax,) = _execute.args_to_matching_eager([argmax], ctx, [_dtypes.int32, _dtypes.int64])
    _attr_T, _inputs_T = _execute.args_to_matching_eager([input, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    input, grad = _inputs_T
    _inputs_flat = [input, grad, argmax]
    _attrs = ('ksize', ksize, 'strides', strides, 'padding', padding, 'include_batch_in_index', include_batch_in_index, 'Targmax', _attr_Targmax, 'T', _attr_T)
    _result = _execute.execute(b'MaxPoolGradWithArgmax', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('MaxPoolGradWithArgmax', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result