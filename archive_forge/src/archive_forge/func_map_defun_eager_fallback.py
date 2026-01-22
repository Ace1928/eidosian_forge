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
def map_defun_eager_fallback(arguments, captured_inputs, output_types, output_shapes, f, max_intra_op_parallelism: int, name, ctx):
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'map_defun' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'map_defun' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if max_intra_op_parallelism is None:
        max_intra_op_parallelism = 1
    max_intra_op_parallelism = _execute.make_int(max_intra_op_parallelism, 'max_intra_op_parallelism')
    _attr_Targuments, arguments = _execute.convert_to_mixed_eager_tensors(arguments, ctx)
    _attr_Tcaptured, captured_inputs = _execute.convert_to_mixed_eager_tensors(captured_inputs, ctx)
    _inputs_flat = list(arguments) + list(captured_inputs)
    _attrs = ('Targuments', _attr_Targuments, 'Tcaptured', _attr_Tcaptured, 'output_types', output_types, 'output_shapes', output_shapes, 'f', f, 'max_intra_op_parallelism', max_intra_op_parallelism)
    _result = _execute.execute(b'MapDefun', len(output_types), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('MapDefun', _inputs_flat, _attrs, _result)
    return _result