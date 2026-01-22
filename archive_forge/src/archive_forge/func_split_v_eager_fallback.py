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
def split_v_eager_fallback(value: _atypes.TensorFuzzingAnnotation[TV_SplitV_T], size_splits: _atypes.TensorFuzzingAnnotation[TV_SplitV_Tlen], axis: _atypes.TensorFuzzingAnnotation[_atypes.Int32], num_split: int, name, ctx):
    num_split = _execute.make_int(num_split, 'num_split')
    _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx, [])
    _attr_Tlen, (size_splits,) = _execute.args_to_matching_eager([size_splits], ctx, [_dtypes.int8, _dtypes.int32, _dtypes.int64], _dtypes.int64)
    axis = _ops.convert_to_tensor(axis, _dtypes.int32)
    _inputs_flat = [value, size_splits, axis]
    _attrs = ('num_split', num_split, 'T', _attr_T, 'Tlen', _attr_Tlen)
    _result = _execute.execute(b'SplitV', num_split, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SplitV', _inputs_flat, _attrs, _result)
    return _result