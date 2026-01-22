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
def kmeans_plus_plus_initialization(points: _atypes.TensorFuzzingAnnotation[_atypes.Float32], num_to_sample: _atypes.TensorFuzzingAnnotation[_atypes.Int64], seed: _atypes.TensorFuzzingAnnotation[_atypes.Int64], num_retries_per_sample: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Selects num_to_sample rows of input using the KMeans++ criterion.

  Rows of points are assumed to be input points. One row is selected at random.
  Subsequent rows are sampled with probability proportional to the squared L2
  distance from the nearest row selected thus far till num_to_sample rows have
  been sampled.

  Args:
    points: A `Tensor` of type `float32`.
      Matrix of shape (n, d). Rows are assumed to be input points.
    num_to_sample: A `Tensor` of type `int64`.
      Scalar. The number of rows to sample. This value must not be larger than n.
    seed: A `Tensor` of type `int64`.
      Scalar. Seed for initializing the random number generator.
    num_retries_per_sample: A `Tensor` of type `int64`.
      Scalar. For each row that is sampled, this parameter
      specifies the number of additional points to draw from the current
      distribution before selecting the best. If a negative value is specified, a
      heuristic is used to sample O(log(num_to_sample)) additional points.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'KmeansPlusPlusInitialization', name, points, num_to_sample, seed, num_retries_per_sample)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return kmeans_plus_plus_initialization_eager_fallback(points, num_to_sample, seed, num_retries_per_sample, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('KmeansPlusPlusInitialization', points=points, num_to_sample=num_to_sample, seed=seed, num_retries_per_sample=num_retries_per_sample, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('KmeansPlusPlusInitialization', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result