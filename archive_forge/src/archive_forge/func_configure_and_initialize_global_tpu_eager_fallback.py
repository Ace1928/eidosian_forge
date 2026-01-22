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
def configure_and_initialize_global_tpu_eager_fallback(use_tfrt_host_runtime: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Int32]:
    if use_tfrt_host_runtime is None:
        use_tfrt_host_runtime = True
    use_tfrt_host_runtime = _execute.make_bool(use_tfrt_host_runtime, 'use_tfrt_host_runtime')
    _inputs_flat = []
    _attrs = ('use_tfrt_host_runtime', use_tfrt_host_runtime)
    _result = _execute.execute(b'ConfigureAndInitializeGlobalTPU', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ConfigureAndInitializeGlobalTPU', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result