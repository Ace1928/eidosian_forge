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
def xla_split_nd_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_XlaSplitND_T], N: int, num_splits, paddings, name, ctx):
    N = _execute.make_int(N, 'N')
    if not isinstance(num_splits, (list, tuple)):
        raise TypeError("Expected list for 'num_splits' argument to 'xla_split_nd' Op, not %r." % num_splits)
    num_splits = [_execute.make_int(_i, 'num_splits') for _i in num_splits]
    if paddings is None:
        paddings = []
    if not isinstance(paddings, (list, tuple)):
        raise TypeError("Expected list for 'paddings' argument to 'xla_split_nd' Op, not %r." % paddings)
    paddings = [_execute.make_int(_i, 'paddings') for _i in paddings]
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
    _inputs_flat = [input]
    _attrs = ('T', _attr_T, 'N', N, 'num_splits', num_splits, 'paddings', paddings)
    _result = _execute.execute(b'XlaSplitND', N, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('XlaSplitND', _inputs_flat, _attrs, _result)
    return _result