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
def compress_element_eager_fallback(components, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    _attr_input_types, components = _execute.convert_to_mixed_eager_tensors(components, ctx)
    _inputs_flat = list(components)
    _attrs = ('input_types', _attr_input_types)
    _result = _execute.execute(b'CompressElement', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CompressElement', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result