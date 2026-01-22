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
def composite_tensor_variant_to_components_eager_fallback(encoded: _atypes.TensorFuzzingAnnotation[_atypes.Variant], metadata: str, Tcomponents, name, ctx):
    metadata = _execute.make_str(metadata, 'metadata')
    if not isinstance(Tcomponents, (list, tuple)):
        raise TypeError("Expected list for 'Tcomponents' argument to 'composite_tensor_variant_to_components' Op, not %r." % Tcomponents)
    Tcomponents = [_execute.make_type(_t, 'Tcomponents') for _t in Tcomponents]
    encoded = _ops.convert_to_tensor(encoded, _dtypes.variant)
    _inputs_flat = [encoded]
    _attrs = ('metadata', metadata, 'Tcomponents', Tcomponents)
    _result = _execute.execute(b'CompositeTensorVariantToComponents', len(Tcomponents), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CompositeTensorVariantToComponents', _inputs_flat, _attrs, _result)
    return _result