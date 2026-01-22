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
def regex_replace_eager_fallback(input: _atypes.TensorFuzzingAnnotation[_atypes.String], pattern: _atypes.TensorFuzzingAnnotation[_atypes.String], rewrite: _atypes.TensorFuzzingAnnotation[_atypes.String], replace_global: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    if replace_global is None:
        replace_global = True
    replace_global = _execute.make_bool(replace_global, 'replace_global')
    input = _ops.convert_to_tensor(input, _dtypes.string)
    pattern = _ops.convert_to_tensor(pattern, _dtypes.string)
    rewrite = _ops.convert_to_tensor(rewrite, _dtypes.string)
    _inputs_flat = [input, pattern, rewrite]
    _attrs = ('replace_global', replace_global)
    _result = _execute.execute(b'RegexReplace', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RegexReplace', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result