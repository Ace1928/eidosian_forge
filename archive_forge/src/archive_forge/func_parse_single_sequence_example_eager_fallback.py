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
def parse_single_sequence_example_eager_fallback(serialized: _atypes.TensorFuzzingAnnotation[_atypes.String], feature_list_dense_missing_assumed_empty: _atypes.TensorFuzzingAnnotation[_atypes.String], context_sparse_keys: List[_atypes.TensorFuzzingAnnotation[_atypes.String]], context_dense_keys: List[_atypes.TensorFuzzingAnnotation[_atypes.String]], feature_list_sparse_keys: List[_atypes.TensorFuzzingAnnotation[_atypes.String]], feature_list_dense_keys: List[_atypes.TensorFuzzingAnnotation[_atypes.String]], context_dense_defaults, debug_name: _atypes.TensorFuzzingAnnotation[_atypes.String], context_sparse_types, feature_list_dense_types, context_dense_shapes, feature_list_sparse_types, feature_list_dense_shapes, name, ctx):
    if not isinstance(context_sparse_keys, (list, tuple)):
        raise TypeError("Expected list for 'context_sparse_keys' argument to 'parse_single_sequence_example' Op, not %r." % context_sparse_keys)
    _attr_Ncontext_sparse = len(context_sparse_keys)
    if not isinstance(context_dense_keys, (list, tuple)):
        raise TypeError("Expected list for 'context_dense_keys' argument to 'parse_single_sequence_example' Op, not %r." % context_dense_keys)
    _attr_Ncontext_dense = len(context_dense_keys)
    if not isinstance(feature_list_sparse_keys, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_sparse_keys' argument to 'parse_single_sequence_example' Op, not %r." % feature_list_sparse_keys)
    _attr_Nfeature_list_sparse = len(feature_list_sparse_keys)
    if not isinstance(feature_list_dense_keys, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_dense_keys' argument to 'parse_single_sequence_example' Op, not %r." % feature_list_dense_keys)
    _attr_Nfeature_list_dense = len(feature_list_dense_keys)
    if context_sparse_types is None:
        context_sparse_types = []
    if not isinstance(context_sparse_types, (list, tuple)):
        raise TypeError("Expected list for 'context_sparse_types' argument to 'parse_single_sequence_example' Op, not %r." % context_sparse_types)
    context_sparse_types = [_execute.make_type(_t, 'context_sparse_types') for _t in context_sparse_types]
    if feature_list_dense_types is None:
        feature_list_dense_types = []
    if not isinstance(feature_list_dense_types, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_dense_types' argument to 'parse_single_sequence_example' Op, not %r." % feature_list_dense_types)
    feature_list_dense_types = [_execute.make_type(_t, 'feature_list_dense_types') for _t in feature_list_dense_types]
    if context_dense_shapes is None:
        context_dense_shapes = []
    if not isinstance(context_dense_shapes, (list, tuple)):
        raise TypeError("Expected list for 'context_dense_shapes' argument to 'parse_single_sequence_example' Op, not %r." % context_dense_shapes)
    context_dense_shapes = [_execute.make_shape(_s, 'context_dense_shapes') for _s in context_dense_shapes]
    if feature_list_sparse_types is None:
        feature_list_sparse_types = []
    if not isinstance(feature_list_sparse_types, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_sparse_types' argument to 'parse_single_sequence_example' Op, not %r." % feature_list_sparse_types)
    feature_list_sparse_types = [_execute.make_type(_t, 'feature_list_sparse_types') for _t in feature_list_sparse_types]
    if feature_list_dense_shapes is None:
        feature_list_dense_shapes = []
    if not isinstance(feature_list_dense_shapes, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_dense_shapes' argument to 'parse_single_sequence_example' Op, not %r." % feature_list_dense_shapes)
    feature_list_dense_shapes = [_execute.make_shape(_s, 'feature_list_dense_shapes') for _s in feature_list_dense_shapes]
    _attr_Tcontext_dense, context_dense_defaults = _execute.convert_to_mixed_eager_tensors(context_dense_defaults, ctx)
    serialized = _ops.convert_to_tensor(serialized, _dtypes.string)
    feature_list_dense_missing_assumed_empty = _ops.convert_to_tensor(feature_list_dense_missing_assumed_empty, _dtypes.string)
    context_sparse_keys = _ops.convert_n_to_tensor(context_sparse_keys, _dtypes.string)
    context_dense_keys = _ops.convert_n_to_tensor(context_dense_keys, _dtypes.string)
    feature_list_sparse_keys = _ops.convert_n_to_tensor(feature_list_sparse_keys, _dtypes.string)
    feature_list_dense_keys = _ops.convert_n_to_tensor(feature_list_dense_keys, _dtypes.string)
    debug_name = _ops.convert_to_tensor(debug_name, _dtypes.string)
    _inputs_flat = [serialized, feature_list_dense_missing_assumed_empty] + list(context_sparse_keys) + list(context_dense_keys) + list(feature_list_sparse_keys) + list(feature_list_dense_keys) + list(context_dense_defaults) + [debug_name]
    _attrs = ('Ncontext_sparse', _attr_Ncontext_sparse, 'Ncontext_dense', _attr_Ncontext_dense, 'Nfeature_list_sparse', _attr_Nfeature_list_sparse, 'Nfeature_list_dense', _attr_Nfeature_list_dense, 'context_sparse_types', context_sparse_types, 'Tcontext_dense', _attr_Tcontext_dense, 'feature_list_dense_types', feature_list_dense_types, 'context_dense_shapes', context_dense_shapes, 'feature_list_sparse_types', feature_list_sparse_types, 'feature_list_dense_shapes', feature_list_dense_shapes)
    _result = _execute.execute(b'ParseSingleSequenceExample', _attr_Ncontext_sparse + len(context_sparse_types) + _attr_Ncontext_sparse + len(context_dense_defaults) + _attr_Nfeature_list_sparse + len(feature_list_sparse_types) + _attr_Nfeature_list_sparse + len(feature_list_dense_types), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ParseSingleSequenceExample', _inputs_flat, _attrs, _result)
    _result = [_result[:_attr_Ncontext_sparse]] + _result[_attr_Ncontext_sparse:]
    _result = _result[:1] + [_result[1:1 + len(context_sparse_types)]] + _result[1 + len(context_sparse_types):]
    _result = _result[:2] + [_result[2:2 + _attr_Ncontext_sparse]] + _result[2 + _attr_Ncontext_sparse:]
    _result = _result[:3] + [_result[3:3 + len(context_dense_defaults)]] + _result[3 + len(context_dense_defaults):]
    _result = _result[:4] + [_result[4:4 + _attr_Nfeature_list_sparse]] + _result[4 + _attr_Nfeature_list_sparse:]
    _result = _result[:5] + [_result[5:5 + len(feature_list_sparse_types)]] + _result[5 + len(feature_list_sparse_types):]
    _result = _result[:6] + [_result[6:6 + _attr_Nfeature_list_sparse]] + _result[6 + _attr_Nfeature_list_sparse:]
    _result = _result[:7] + [_result[7:]]
    _result = _ParseSingleSequenceExampleOutput._make(_result)
    return _result