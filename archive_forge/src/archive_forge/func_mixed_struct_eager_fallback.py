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
def mixed_struct_eager_fallback(n_a: int, name, ctx):
    n_a = _execute.make_int(n_a, 'n_a')
    _inputs_flat = []
    _attrs = ('n_a', n_a)
    _result = _execute.execute(b'MixedStruct', n_a + 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('MixedStruct', _inputs_flat, _attrs, _result)
    _result = [_result[:n_a]] + _result[n_a:]
    _result = _MixedStructOutput._make(_result)
    return _result