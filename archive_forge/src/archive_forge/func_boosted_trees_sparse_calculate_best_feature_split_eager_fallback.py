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
def boosted_trees_sparse_calculate_best_feature_split_eager_fallback(node_id_range: _atypes.TensorFuzzingAnnotation[_atypes.Int32], stats_summary_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int32], stats_summary_values: _atypes.TensorFuzzingAnnotation[_atypes.Float32], stats_summary_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int32], l1: _atypes.TensorFuzzingAnnotation[_atypes.Float32], l2: _atypes.TensorFuzzingAnnotation[_atypes.Float32], tree_complexity: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_node_weight: _atypes.TensorFuzzingAnnotation[_atypes.Float32], logits_dimension: int, split_type: str, name, ctx):
    logits_dimension = _execute.make_int(logits_dimension, 'logits_dimension')
    if split_type is None:
        split_type = 'inequality'
    split_type = _execute.make_str(split_type, 'split_type')
    node_id_range = _ops.convert_to_tensor(node_id_range, _dtypes.int32)
    stats_summary_indices = _ops.convert_to_tensor(stats_summary_indices, _dtypes.int32)
    stats_summary_values = _ops.convert_to_tensor(stats_summary_values, _dtypes.float32)
    stats_summary_shape = _ops.convert_to_tensor(stats_summary_shape, _dtypes.int32)
    l1 = _ops.convert_to_tensor(l1, _dtypes.float32)
    l2 = _ops.convert_to_tensor(l2, _dtypes.float32)
    tree_complexity = _ops.convert_to_tensor(tree_complexity, _dtypes.float32)
    min_node_weight = _ops.convert_to_tensor(min_node_weight, _dtypes.float32)
    _inputs_flat = [node_id_range, stats_summary_indices, stats_summary_values, stats_summary_shape, l1, l2, tree_complexity, min_node_weight]
    _attrs = ('logits_dimension', logits_dimension, 'split_type', split_type)
    _result = _execute.execute(b'BoostedTreesSparseCalculateBestFeatureSplit', 7, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BoostedTreesSparseCalculateBestFeatureSplit', _inputs_flat, _attrs, _result)
    _result = _BoostedTreesSparseCalculateBestFeatureSplitOutput._make(_result)
    return _result