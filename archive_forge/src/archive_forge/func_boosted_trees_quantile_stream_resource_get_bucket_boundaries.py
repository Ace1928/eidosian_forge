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
def boosted_trees_quantile_stream_resource_get_bucket_boundaries(quantile_stream_resource_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], num_features: int, name=None):
    """Generate the bucket boundaries for each feature based on accumulated summaries.

  An op that returns a list of float tensors for a quantile stream resource. Each
  tensor is Rank 1 containing bucket boundaries for a single feature.

  Args:
    quantile_stream_resource_handle: A `Tensor` of type `resource`.
      resource handle referring to a QuantileStreamResource.
    num_features: An `int` that is `>= 0`.
      inferred int; number of features to get bucket boundaries for.
    name: A name for the operation (optional).

  Returns:
    A list of `num_features` `Tensor` objects with type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesQuantileStreamResourceGetBucketBoundaries', name, quantile_stream_resource_handle, 'num_features', num_features)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_quantile_stream_resource_get_bucket_boundaries_eager_fallback(quantile_stream_resource_handle, num_features=num_features, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    num_features = _execute.make_int(num_features, 'num_features')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesQuantileStreamResourceGetBucketBoundaries', quantile_stream_resource_handle=quantile_stream_resource_handle, num_features=num_features, name=name)
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('num_features', _op._get_attr_int('num_features'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BoostedTreesQuantileStreamResourceGetBucketBoundaries', _inputs_flat, _attrs, _result)
    return _result