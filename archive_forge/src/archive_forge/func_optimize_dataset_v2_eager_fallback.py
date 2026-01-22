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
def optimize_dataset_v2_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], optimizations_enabled: _atypes.TensorFuzzingAnnotation[_atypes.String], optimizations_disabled: _atypes.TensorFuzzingAnnotation[_atypes.String], optimizations_default: _atypes.TensorFuzzingAnnotation[_atypes.String], output_types, output_shapes, optimization_configs, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'optimize_dataset_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'optimize_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if optimization_configs is None:
        optimization_configs = []
    if not isinstance(optimization_configs, (list, tuple)):
        raise TypeError("Expected list for 'optimization_configs' argument to 'optimize_dataset_v2' Op, not %r." % optimization_configs)
    optimization_configs = [_execute.make_str(_s, 'optimization_configs') for _s in optimization_configs]
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    optimizations_enabled = _ops.convert_to_tensor(optimizations_enabled, _dtypes.string)
    optimizations_disabled = _ops.convert_to_tensor(optimizations_disabled, _dtypes.string)
    optimizations_default = _ops.convert_to_tensor(optimizations_default, _dtypes.string)
    _inputs_flat = [input_dataset, optimizations_enabled, optimizations_disabled, optimizations_default]
    _attrs = ('output_types', output_types, 'output_shapes', output_shapes, 'optimization_configs', optimization_configs)
    _result = _execute.execute(b'OptimizeDatasetV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('OptimizeDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result