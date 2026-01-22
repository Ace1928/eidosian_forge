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
def infeed_enqueue_tuple_eager_fallback(inputs, shapes, layouts, device_ordinal: int, name, ctx):
    if not isinstance(shapes, (list, tuple)):
        raise TypeError("Expected list for 'shapes' argument to 'infeed_enqueue_tuple' Op, not %r." % shapes)
    shapes = [_execute.make_shape(_s, 'shapes') for _s in shapes]
    if layouts is None:
        layouts = []
    if not isinstance(layouts, (list, tuple)):
        raise TypeError("Expected list for 'layouts' argument to 'infeed_enqueue_tuple' Op, not %r." % layouts)
    layouts = [_execute.make_int(_i, 'layouts') for _i in layouts]
    if device_ordinal is None:
        device_ordinal = -1
    device_ordinal = _execute.make_int(device_ordinal, 'device_ordinal')
    _attr_dtypes, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
    _inputs_flat = list(inputs)
    _attrs = ('dtypes', _attr_dtypes, 'shapes', shapes, 'layouts', layouts, 'device_ordinal', device_ordinal)
    _result = _execute.execute(b'InfeedEnqueueTuple', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result