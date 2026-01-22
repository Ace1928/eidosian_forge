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
def parse_example_dataset_v2_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], num_parallel_calls: _atypes.TensorFuzzingAnnotation[_atypes.Int64], dense_defaults, sparse_keys, dense_keys, sparse_types, dense_shapes, output_types, output_shapes, deterministic: str, ragged_keys, ragged_value_types, ragged_split_types, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(sparse_keys, (list, tuple)):
        raise TypeError("Expected list for 'sparse_keys' argument to 'parse_example_dataset_v2' Op, not %r." % sparse_keys)
    sparse_keys = [_execute.make_str(_s, 'sparse_keys') for _s in sparse_keys]
    if not isinstance(dense_keys, (list, tuple)):
        raise TypeError("Expected list for 'dense_keys' argument to 'parse_example_dataset_v2' Op, not %r." % dense_keys)
    dense_keys = [_execute.make_str(_s, 'dense_keys') for _s in dense_keys]
    if not isinstance(sparse_types, (list, tuple)):
        raise TypeError("Expected list for 'sparse_types' argument to 'parse_example_dataset_v2' Op, not %r." % sparse_types)
    sparse_types = [_execute.make_type(_t, 'sparse_types') for _t in sparse_types]
    if not isinstance(dense_shapes, (list, tuple)):
        raise TypeError("Expected list for 'dense_shapes' argument to 'parse_example_dataset_v2' Op, not %r." % dense_shapes)
    dense_shapes = [_execute.make_shape(_s, 'dense_shapes') for _s in dense_shapes]
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'parse_example_dataset_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'parse_example_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if deterministic is None:
        deterministic = 'default'
    deterministic = _execute.make_str(deterministic, 'deterministic')
    if ragged_keys is None:
        ragged_keys = []
    if not isinstance(ragged_keys, (list, tuple)):
        raise TypeError("Expected list for 'ragged_keys' argument to 'parse_example_dataset_v2' Op, not %r." % ragged_keys)
    ragged_keys = [_execute.make_str(_s, 'ragged_keys') for _s in ragged_keys]
    if ragged_value_types is None:
        ragged_value_types = []
    if not isinstance(ragged_value_types, (list, tuple)):
        raise TypeError("Expected list for 'ragged_value_types' argument to 'parse_example_dataset_v2' Op, not %r." % ragged_value_types)
    ragged_value_types = [_execute.make_type(_t, 'ragged_value_types') for _t in ragged_value_types]
    if ragged_split_types is None:
        ragged_split_types = []
    if not isinstance(ragged_split_types, (list, tuple)):
        raise TypeError("Expected list for 'ragged_split_types' argument to 'parse_example_dataset_v2' Op, not %r." % ragged_split_types)
    ragged_split_types = [_execute.make_type(_t, 'ragged_split_types') for _t in ragged_split_types]
    _attr_Tdense, dense_defaults = _execute.convert_to_mixed_eager_tensors(dense_defaults, ctx)
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
    _inputs_flat = [input_dataset, num_parallel_calls] + list(dense_defaults)
    _attrs = ('sparse_keys', sparse_keys, 'dense_keys', dense_keys, 'sparse_types', sparse_types, 'Tdense', _attr_Tdense, 'dense_shapes', dense_shapes, 'output_types', output_types, 'output_shapes', output_shapes, 'deterministic', deterministic, 'ragged_keys', ragged_keys, 'ragged_value_types', ragged_value_types, 'ragged_split_types', ragged_split_types)
    _result = _execute.execute(b'ParseExampleDatasetV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ParseExampleDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result