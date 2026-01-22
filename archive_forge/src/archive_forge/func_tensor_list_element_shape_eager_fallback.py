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
def tensor_list_element_shape_eager_fallback(input_handle: _atypes.TensorFuzzingAnnotation[_atypes.Variant], shape_type: TV_TensorListElementShape_shape_type, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_TensorListElementShape_shape_type]:
    shape_type = _execute.make_type(shape_type, 'shape_type')
    input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
    _inputs_flat = [input_handle]
    _attrs = ('shape_type', shape_type)
    _result = _execute.execute(b'TensorListElementShape', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TensorListElementShape', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result