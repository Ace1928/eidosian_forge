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
def boosted_trees_example_debug_outputs_eager_fallback(tree_ensemble_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], bucketized_features: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], logits_dimension: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    if not isinstance(bucketized_features, (list, tuple)):
        raise TypeError("Expected list for 'bucketized_features' argument to 'boosted_trees_example_debug_outputs' Op, not %r." % bucketized_features)
    _attr_num_bucketized_features = len(bucketized_features)
    logits_dimension = _execute.make_int(logits_dimension, 'logits_dimension')
    tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
    bucketized_features = _ops.convert_n_to_tensor(bucketized_features, _dtypes.int32)
    _inputs_flat = [tree_ensemble_handle] + list(bucketized_features)
    _attrs = ('num_bucketized_features', _attr_num_bucketized_features, 'logits_dimension', logits_dimension)
    _result = _execute.execute(b'BoostedTreesExampleDebugOutputs', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BoostedTreesExampleDebugOutputs', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result