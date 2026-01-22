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
def parse_sequence_example_eager_fallback(serialized: _atypes.TensorFuzzingAnnotation[_atypes.String], debug_name: _atypes.TensorFuzzingAnnotation[_atypes.String], context_dense_defaults, feature_list_dense_missing_assumed_empty, context_sparse_keys, context_dense_keys, feature_list_sparse_keys, feature_list_dense_keys, Ncontext_sparse: int, Ncontext_dense: int, Nfeature_list_sparse: int, Nfeature_list_dense: int, context_sparse_types, feature_list_dense_types, context_dense_shapes, feature_list_sparse_types, feature_list_dense_shapes, name, ctx):
    if not isinstance(feature_list_dense_missing_assumed_empty, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_dense_missing_assumed_empty' argument to 'parse_sequence_example' Op, not %r." % feature_list_dense_missing_assumed_empty)
    feature_list_dense_missing_assumed_empty = [_execute.make_str(_s, 'feature_list_dense_missing_assumed_empty') for _s in feature_list_dense_missing_assumed_empty]
    if not isinstance(context_sparse_keys, (list, tuple)):
        raise TypeError("Expected list for 'context_sparse_keys' argument to 'parse_sequence_example' Op, not %r." % context_sparse_keys)
    context_sparse_keys = [_execute.make_str(_s, 'context_sparse_keys') for _s in context_sparse_keys]
    if not isinstance(context_dense_keys, (list, tuple)):
        raise TypeError("Expected list for 'context_dense_keys' argument to 'parse_sequence_example' Op, not %r." % context_dense_keys)
    context_dense_keys = [_execute.make_str(_s, 'context_dense_keys') for _s in context_dense_keys]
    if not isinstance(feature_list_sparse_keys, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_sparse_keys' argument to 'parse_sequence_example' Op, not %r." % feature_list_sparse_keys)
    feature_list_sparse_keys = [_execute.make_str(_s, 'feature_list_sparse_keys') for _s in feature_list_sparse_keys]
    if not isinstance(feature_list_dense_keys, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_dense_keys' argument to 'parse_sequence_example' Op, not %r." % feature_list_dense_keys)
    feature_list_dense_keys = [_execute.make_str(_s, 'feature_list_dense_keys') for _s in feature_list_dense_keys]
    if Ncontext_sparse is None:
        Ncontext_sparse = 0
    Ncontext_sparse = _execute.make_int(Ncontext_sparse, 'Ncontext_sparse')
    if Ncontext_dense is None:
        Ncontext_dense = 0
    Ncontext_dense = _execute.make_int(Ncontext_dense, 'Ncontext_dense')
    if Nfeature_list_sparse is None:
        Nfeature_list_sparse = 0
    Nfeature_list_sparse = _execute.make_int(Nfeature_list_sparse, 'Nfeature_list_sparse')
    if Nfeature_list_dense is None:
        Nfeature_list_dense = 0
    Nfeature_list_dense = _execute.make_int(Nfeature_list_dense, 'Nfeature_list_dense')
    if context_sparse_types is None:
        context_sparse_types = []
    if not isinstance(context_sparse_types, (list, tuple)):
        raise TypeError("Expected list for 'context_sparse_types' argument to 'parse_sequence_example' Op, not %r." % context_sparse_types)
    context_sparse_types = [_execute.make_type(_t, 'context_sparse_types') for _t in context_sparse_types]
    if feature_list_dense_types is None:
        feature_list_dense_types = []
    if not isinstance(feature_list_dense_types, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_dense_types' argument to 'parse_sequence_example' Op, not %r." % feature_list_dense_types)
    feature_list_dense_types = [_execute.make_type(_t, 'feature_list_dense_types') for _t in feature_list_dense_types]
    if context_dense_shapes is None:
        context_dense_shapes = []
    if not isinstance(context_dense_shapes, (list, tuple)):
        raise TypeError("Expected list for 'context_dense_shapes' argument to 'parse_sequence_example' Op, not %r." % context_dense_shapes)
    context_dense_shapes = [_execute.make_shape(_s, 'context_dense_shapes') for _s in context_dense_shapes]
    if feature_list_sparse_types is None:
        feature_list_sparse_types = []
    if not isinstance(feature_list_sparse_types, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_sparse_types' argument to 'parse_sequence_example' Op, not %r." % feature_list_sparse_types)
    feature_list_sparse_types = [_execute.make_type(_t, 'feature_list_sparse_types') for _t in feature_list_sparse_types]
    if feature_list_dense_shapes is None:
        feature_list_dense_shapes = []
    if not isinstance(feature_list_dense_shapes, (list, tuple)):
        raise TypeError("Expected list for 'feature_list_dense_shapes' argument to 'parse_sequence_example' Op, not %r." % feature_list_dense_shapes)
    feature_list_dense_shapes = [_execute.make_shape(_s, 'feature_list_dense_shapes') for _s in feature_list_dense_shapes]
    _attr_Tcontext_dense, context_dense_defaults = _execute.convert_to_mixed_eager_tensors(context_dense_defaults, ctx)
    serialized = _ops.convert_to_tensor(serialized, _dtypes.string)
    debug_name = _ops.convert_to_tensor(debug_name, _dtypes.string)
    _inputs_flat = [serialized, debug_name] + list(context_dense_defaults)
    _attrs = ('feature_list_dense_missing_assumed_empty', feature_list_dense_missing_assumed_empty, 'context_sparse_keys', context_sparse_keys, 'context_dense_keys', context_dense_keys, 'feature_list_sparse_keys', feature_list_sparse_keys, 'feature_list_dense_keys', feature_list_dense_keys, 'Ncontext_sparse', Ncontext_sparse, 'Ncontext_dense', Ncontext_dense, 'Nfeature_list_sparse', Nfeature_list_sparse, 'Nfeature_list_dense', Nfeature_list_dense, 'context_sparse_types', context_sparse_types, 'Tcontext_dense', _attr_Tcontext_dense, 'feature_list_dense_types', feature_list_dense_types, 'context_dense_shapes', context_dense_shapes, 'feature_list_sparse_types', feature_list_sparse_types, 'feature_list_dense_shapes', feature_list_dense_shapes)
    _result = _execute.execute(b'ParseSequenceExample', Ncontext_sparse + len(context_sparse_types) + Ncontext_sparse + len(context_dense_defaults) + Nfeature_list_sparse + len(feature_list_sparse_types) + Nfeature_list_sparse + len(feature_list_dense_types) + Nfeature_list_dense, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ParseSequenceExample', _inputs_flat, _attrs, _result)
    _result = [_result[:Ncontext_sparse]] + _result[Ncontext_sparse:]
    _result = _result[:1] + [_result[1:1 + len(context_sparse_types)]] + _result[1 + len(context_sparse_types):]
    _result = _result[:2] + [_result[2:2 + Ncontext_sparse]] + _result[2 + Ncontext_sparse:]
    _result = _result[:3] + [_result[3:3 + len(context_dense_defaults)]] + _result[3 + len(context_dense_defaults):]
    _result = _result[:4] + [_result[4:4 + Nfeature_list_sparse]] + _result[4 + Nfeature_list_sparse:]
    _result = _result[:5] + [_result[5:5 + len(feature_list_sparse_types)]] + _result[5 + len(feature_list_sparse_types):]
    _result = _result[:6] + [_result[6:6 + Nfeature_list_sparse]] + _result[6 + Nfeature_list_sparse:]
    _result = _result[:7] + [_result[7:7 + len(feature_list_dense_types)]] + _result[7 + len(feature_list_dense_types):]
    _result = _result[:8] + [_result[8:]]
    _result = _ParseSequenceExampleOutput._make(_result)
    return _result