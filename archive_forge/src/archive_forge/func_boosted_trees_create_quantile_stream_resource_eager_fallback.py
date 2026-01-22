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
def boosted_trees_create_quantile_stream_resource_eager_fallback(quantile_stream_resource_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], epsilon: _atypes.TensorFuzzingAnnotation[_atypes.Float32], num_streams: _atypes.TensorFuzzingAnnotation[_atypes.Int64], max_elements: int, name, ctx):
    if max_elements is None:
        max_elements = 1099511627776
    max_elements = _execute.make_int(max_elements, 'max_elements')
    quantile_stream_resource_handle = _ops.convert_to_tensor(quantile_stream_resource_handle, _dtypes.resource)
    epsilon = _ops.convert_to_tensor(epsilon, _dtypes.float32)
    num_streams = _ops.convert_to_tensor(num_streams, _dtypes.int64)
    _inputs_flat = [quantile_stream_resource_handle, epsilon, num_streams]
    _attrs = ('max_elements', max_elements)
    _result = _execute.execute(b'BoostedTreesCreateQuantileStreamResource', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result