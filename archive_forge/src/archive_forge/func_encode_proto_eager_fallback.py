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
def encode_proto_eager_fallback(sizes: _atypes.TensorFuzzingAnnotation[_atypes.Int32], values, field_names, message_type: str, descriptor_source: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    if not isinstance(field_names, (list, tuple)):
        raise TypeError("Expected list for 'field_names' argument to 'encode_proto' Op, not %r." % field_names)
    field_names = [_execute.make_str(_s, 'field_names') for _s in field_names]
    message_type = _execute.make_str(message_type, 'message_type')
    if descriptor_source is None:
        descriptor_source = 'local://'
    descriptor_source = _execute.make_str(descriptor_source, 'descriptor_source')
    _attr_Tinput_types, values = _execute.convert_to_mixed_eager_tensors(values, ctx)
    sizes = _ops.convert_to_tensor(sizes, _dtypes.int32)
    _inputs_flat = [sizes] + list(values)
    _attrs = ('field_names', field_names, 'message_type', message_type, 'descriptor_source', descriptor_source, 'Tinput_types', _attr_Tinput_types)
    _result = _execute.execute(b'EncodeProto', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('EncodeProto', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result