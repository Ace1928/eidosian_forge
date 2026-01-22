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
def dtype_with_default_op_eager_fallback(in_: _atypes.TensorFuzzingAnnotation[TV_DtypeWithDefaultOp_T], name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    _attr_T, (in_,) = _execute.args_to_matching_eager([in_], ctx, [], _dtypes.uint8)
    _inputs_flat = [in_]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'DtypeWithDefaultOp', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DtypeWithDefaultOp', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result