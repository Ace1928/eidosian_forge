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
def boosted_trees_quantile_stream_resource_get_bucket_boundaries_eager_fallback(quantile_stream_resource_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], num_features: int, name, ctx):
    num_features = _execute.make_int(num_features, 'num_features')
    quantile_stream_resource_handle = _ops.convert_to_tensor(quantile_stream_resource_handle, _dtypes.resource)
    _inputs_flat = [quantile_stream_resource_handle]
    _attrs = ('num_features', num_features)
    _result = _execute.execute(b'BoostedTreesQuantileStreamResourceGetBucketBoundaries', num_features, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BoostedTreesQuantileStreamResourceGetBucketBoundaries', _inputs_flat, _attrs, _result)
    return _result