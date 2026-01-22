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
def stateless_case_eager_fallback(branch_index: _atypes.TensorFuzzingAnnotation[_atypes.Int32], input, Tout, branches, output_shapes, name, ctx):
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'stateless_case' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    if not isinstance(branches, (list, tuple)):
        raise TypeError("Expected list for 'branches' argument to 'stateless_case' Op, not %r." % branches)
    if output_shapes is None:
        output_shapes = []
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'stateless_case' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    _attr_Tin, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
    branch_index = _ops.convert_to_tensor(branch_index, _dtypes.int32)
    _inputs_flat = [branch_index] + list(input)
    _attrs = ('Tin', _attr_Tin, 'Tout', Tout, 'branches', branches, 'output_shapes', output_shapes)
    _result = _execute.execute(b'StatelessCase', len(Tout), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('StatelessCase', _inputs_flat, _attrs, _result)
    return _result