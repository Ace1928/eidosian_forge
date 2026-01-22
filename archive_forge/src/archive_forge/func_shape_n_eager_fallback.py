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
def shape_n_eager_fallback(input: List[_atypes.TensorFuzzingAnnotation[TV_ShapeN_T]], out_type: TV_ShapeN_out_type, name, ctx):
    if not isinstance(input, (list, tuple)):
        raise TypeError("Expected list for 'input' argument to 'shape_n' Op, not %r." % input)
    _attr_N = len(input)
    if out_type is None:
        out_type = _dtypes.int32
    out_type = _execute.make_type(out_type, 'out_type')
    _attr_T, input = _execute.args_to_matching_eager(list(input), ctx, [])
    _inputs_flat = list(input)
    _attrs = ('N', _attr_N, 'T', _attr_T, 'out_type', out_type)
    _result = _execute.execute(b'ShapeN', _attr_N, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ShapeN', _inputs_flat, _attrs, _result)
    return _result