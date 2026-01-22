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
def print_v2_eager_fallback(input: _atypes.TensorFuzzingAnnotation[_atypes.String], output_stream: str, end: str, name, ctx):
    if output_stream is None:
        output_stream = 'stderr'
    output_stream = _execute.make_str(output_stream, 'output_stream')
    if end is None:
        end = '\n'
    end = _execute.make_str(end, 'end')
    input = _ops.convert_to_tensor(input, _dtypes.string)
    _inputs_flat = [input]
    _attrs = ('output_stream', output_stream, 'end', end)
    _result = _execute.execute(b'PrintV2', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result