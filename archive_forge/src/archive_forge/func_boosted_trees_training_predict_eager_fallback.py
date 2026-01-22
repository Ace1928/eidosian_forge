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
def boosted_trees_training_predict_eager_fallback(tree_ensemble_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], cached_tree_ids: _atypes.TensorFuzzingAnnotation[_atypes.Int32], cached_node_ids: _atypes.TensorFuzzingAnnotation[_atypes.Int32], bucketized_features: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], logits_dimension: int, name, ctx):
    if not isinstance(bucketized_features, (list, tuple)):
        raise TypeError("Expected list for 'bucketized_features' argument to 'boosted_trees_training_predict' Op, not %r." % bucketized_features)
    _attr_num_bucketized_features = len(bucketized_features)
    logits_dimension = _execute.make_int(logits_dimension, 'logits_dimension')
    tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
    cached_tree_ids = _ops.convert_to_tensor(cached_tree_ids, _dtypes.int32)
    cached_node_ids = _ops.convert_to_tensor(cached_node_ids, _dtypes.int32)
    bucketized_features = _ops.convert_n_to_tensor(bucketized_features, _dtypes.int32)
    _inputs_flat = [tree_ensemble_handle, cached_tree_ids, cached_node_ids] + list(bucketized_features)
    _attrs = ('num_bucketized_features', _attr_num_bucketized_features, 'logits_dimension', logits_dimension)
    _result = _execute.execute(b'BoostedTreesTrainingPredict', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BoostedTreesTrainingPredict', _inputs_flat, _attrs, _result)
    _result = _BoostedTreesTrainingPredictOutput._make(_result)
    return _result