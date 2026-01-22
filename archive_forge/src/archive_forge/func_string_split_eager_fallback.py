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
def string_split_eager_fallback(input: _atypes.TensorFuzzingAnnotation[_atypes.String], delimiter: _atypes.TensorFuzzingAnnotation[_atypes.String], skip_empty: bool, name, ctx):
    if skip_empty is None:
        skip_empty = True
    skip_empty = _execute.make_bool(skip_empty, 'skip_empty')
    input = _ops.convert_to_tensor(input, _dtypes.string)
    delimiter = _ops.convert_to_tensor(delimiter, _dtypes.string)
    _inputs_flat = [input, delimiter]
    _attrs = ('skip_empty', skip_empty)
    _result = _execute.execute(b'StringSplit', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('StringSplit', _inputs_flat, _attrs, _result)
    _result = _StringSplitOutput._make(_result)
    return _result