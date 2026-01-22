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
def assign_variable_xla_concat_nd_eager_fallback(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], inputs: List[_atypes.TensorFuzzingAnnotation[TV_AssignVariableXlaConcatND_T]], num_concats, paddings, name, ctx):
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Expected list for 'inputs' argument to 'assign_variable_xla_concat_nd' Op, not %r." % inputs)
    _attr_N = len(inputs)
    if not isinstance(num_concats, (list, tuple)):
        raise TypeError("Expected list for 'num_concats' argument to 'assign_variable_xla_concat_nd' Op, not %r." % num_concats)
    num_concats = [_execute.make_int(_i, 'num_concats') for _i in num_concats]
    if paddings is None:
        paddings = []
    if not isinstance(paddings, (list, tuple)):
        raise TypeError("Expected list for 'paddings' argument to 'assign_variable_xla_concat_nd' Op, not %r." % paddings)
    paddings = [_execute.make_int(_i, 'paddings') for _i in paddings]
    _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), ctx, [])
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource] + list(inputs)
    _attrs = ('T', _attr_T, 'N', _attr_N, 'num_concats', num_concats, 'paddings', paddings)
    _result = _execute.execute(b'AssignVariableXlaConcatND', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result