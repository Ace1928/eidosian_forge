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
def attr_empty_list_default_eager_fallback(a, name, ctx):
    if a is None:
        a = []
    if not isinstance(a, (list, tuple)):
        raise TypeError("Expected list for 'a' argument to 'attr_empty_list_default' Op, not %r." % a)
    a = [_execute.make_float(_f, 'a') for _f in a]
    _inputs_flat = []
    _attrs = ('a', a)
    _result = _execute.execute(b'AttrEmptyListDefault', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result