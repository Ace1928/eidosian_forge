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
def boosted_trees_center_bias_eager_fallback(tree_ensemble_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], mean_gradients: _atypes.TensorFuzzingAnnotation[_atypes.Float32], mean_hessians: _atypes.TensorFuzzingAnnotation[_atypes.Float32], l1: _atypes.TensorFuzzingAnnotation[_atypes.Float32], l2: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Bool]:
    tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
    mean_gradients = _ops.convert_to_tensor(mean_gradients, _dtypes.float32)
    mean_hessians = _ops.convert_to_tensor(mean_hessians, _dtypes.float32)
    l1 = _ops.convert_to_tensor(l1, _dtypes.float32)
    l2 = _ops.convert_to_tensor(l2, _dtypes.float32)
    _inputs_flat = [tree_ensemble_handle, mean_gradients, mean_hessians, l1, l2]
    _attrs = None
    _result = _execute.execute(b'BoostedTreesCenterBias', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BoostedTreesCenterBias', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result