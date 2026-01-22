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
def var_handle_op_eager_fallback(dtype: TV_VarHandleOp_dtype, shape, container: str, shared_name: str, allowed_devices, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    dtype = _execute.make_type(dtype, 'dtype')
    shape = _execute.make_shape(shape, 'shape')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    if allowed_devices is None:
        allowed_devices = []
    if not isinstance(allowed_devices, (list, tuple)):
        raise TypeError("Expected list for 'allowed_devices' argument to 'var_handle_op' Op, not %r." % allowed_devices)
    allowed_devices = [_execute.make_str(_s, 'allowed_devices') for _s in allowed_devices]
    _inputs_flat = []
    _attrs = ('container', container, 'shared_name', shared_name, 'dtype', dtype, 'shape', shape, 'allowed_devices', allowed_devices)
    _result = _execute.execute(b'VarHandleOp', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('VarHandleOp', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result