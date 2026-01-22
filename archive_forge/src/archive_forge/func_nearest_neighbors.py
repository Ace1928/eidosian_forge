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
def nearest_neighbors(points: _atypes.TensorFuzzingAnnotation[_atypes.Float32], centers: _atypes.TensorFuzzingAnnotation[_atypes.Float32], k: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name=None):
    """Selects the k nearest centers for each point.

  Rows of points are assumed to be input points. Rows of centers are assumed to be
  the list of candidate centers. For each point, the k centers that have least L2
  distance to it are computed.

  Args:
    points: A `Tensor` of type `float32`.
      Matrix of shape (n, d). Rows are assumed to be input points.
    centers: A `Tensor` of type `float32`.
      Matrix of shape (m, d). Rows are assumed to be centers.
    k: A `Tensor` of type `int64`.
      Number of nearest centers to return for each point. If k is larger than m, then
      only m centers are returned.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (nearest_center_indices, nearest_center_distances).

    nearest_center_indices: A `Tensor` of type `int64`.
    nearest_center_distances: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'NearestNeighbors', name, points, centers, k)
            _result = _NearestNeighborsOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return nearest_neighbors_eager_fallback(points, centers, k, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('NearestNeighbors', points=points, centers=centers, k=k, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('NearestNeighbors', _inputs_flat, _attrs, _result)
    _result = _NearestNeighborsOutput._make(_result)
    return _result