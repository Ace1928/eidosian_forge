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
def boosted_trees_calculate_best_feature_split_v2_eager_fallback(node_id_range: _atypes.TensorFuzzingAnnotation[_atypes.Int32], stats_summaries_list: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], split_types: _atypes.TensorFuzzingAnnotation[_atypes.String], candidate_feature_ids: _atypes.TensorFuzzingAnnotation[_atypes.Int32], l1: _atypes.TensorFuzzingAnnotation[_atypes.Float32], l2: _atypes.TensorFuzzingAnnotation[_atypes.Float32], tree_complexity: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_node_weight: _atypes.TensorFuzzingAnnotation[_atypes.Float32], logits_dimension: int, name, ctx):
    if not isinstance(stats_summaries_list, (list, tuple)):
        raise TypeError("Expected list for 'stats_summaries_list' argument to 'boosted_trees_calculate_best_feature_split_v2' Op, not %r." % stats_summaries_list)
    _attr_num_features = len(stats_summaries_list)
    logits_dimension = _execute.make_int(logits_dimension, 'logits_dimension')
    node_id_range = _ops.convert_to_tensor(node_id_range, _dtypes.int32)
    stats_summaries_list = _ops.convert_n_to_tensor(stats_summaries_list, _dtypes.float32)
    split_types = _ops.convert_to_tensor(split_types, _dtypes.string)
    candidate_feature_ids = _ops.convert_to_tensor(candidate_feature_ids, _dtypes.int32)
    l1 = _ops.convert_to_tensor(l1, _dtypes.float32)
    l2 = _ops.convert_to_tensor(l2, _dtypes.float32)
    tree_complexity = _ops.convert_to_tensor(tree_complexity, _dtypes.float32)
    min_node_weight = _ops.convert_to_tensor(min_node_weight, _dtypes.float32)
    _inputs_flat = [node_id_range] + list(stats_summaries_list) + [split_types, candidate_feature_ids, l1, l2, tree_complexity, min_node_weight]
    _attrs = ('num_features', _attr_num_features, 'logits_dimension', logits_dimension)
    _result = _execute.execute(b'BoostedTreesCalculateBestFeatureSplitV2', 8, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BoostedTreesCalculateBestFeatureSplitV2', _inputs_flat, _attrs, _result)
    _result = _BoostedTreesCalculateBestFeatureSplitV2Output._make(_result)
    return _result