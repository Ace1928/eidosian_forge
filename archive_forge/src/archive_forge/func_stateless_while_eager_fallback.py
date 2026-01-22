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
def stateless_while_eager_fallback(input, cond, body, output_shapes, parallel_iterations: int, name, ctx):
    if output_shapes is None:
        output_shapes = []
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'stateless_while' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if parallel_iterations is None:
        parallel_iterations = 10
    parallel_iterations = _execute.make_int(parallel_iterations, 'parallel_iterations')
    _attr_T, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
    _inputs_flat = list(input)
    _attrs = ('T', _attr_T, 'cond', cond, 'body', body, 'output_shapes', output_shapes, 'parallel_iterations', parallel_iterations)
    _result = _execute.execute(b'StatelessWhile', len(input), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('StatelessWhile', _inputs_flat, _attrs, _result)
    return _result