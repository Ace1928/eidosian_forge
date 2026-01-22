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
def sliding_window_dataset_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], window_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], window_shift: _atypes.TensorFuzzingAnnotation[_atypes.Int64], window_stride: _atypes.TensorFuzzingAnnotation[_atypes.Int64], output_types, output_shapes, drop_remainder: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'sliding_window_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'sliding_window_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if drop_remainder is None:
        drop_remainder = True
    drop_remainder = _execute.make_bool(drop_remainder, 'drop_remainder')
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    window_size = _ops.convert_to_tensor(window_size, _dtypes.int64)
    window_shift = _ops.convert_to_tensor(window_shift, _dtypes.int64)
    window_stride = _ops.convert_to_tensor(window_stride, _dtypes.int64)
    _inputs_flat = [input_dataset, window_size, window_shift, window_stride]
    _attrs = ('drop_remainder', drop_remainder, 'output_types', output_types, 'output_shapes', output_shapes)
    _result = _execute.execute(b'SlidingWindowDataset', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SlidingWindowDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result