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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_svd')
def xla_svd(a: _atypes.TensorFuzzingAnnotation[TV_XlaSvd_T], max_iter: int, epsilon: float, precision_config: str, name=None):
    """Computes the eigen decomposition of a batch of self-adjoint matrices

  (Note: Only real inputs are supported).

  Computes the eigenvalues and eigenvectors of the innermost M-by-N matrices in
  tensor such that tensor[...,:,:] = u[..., :, :] * Diag(s[..., :]) * Transpose(v[...,:,:]).

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the input tensor.
    max_iter: An `int`.
      maximum number of sweep update, i.e., the whole lower triangular
      part or upper triangular part based on parameter lower. Heuristically, it has
      been argued that approximately log(min (M, N)) sweeps are needed in practice
      (Ref: Golub & van Loan "Matrix Computation").
    epsilon: A `float`. the tolerance ratio.
    precision_config: A `string`. a serialized xla::PrecisionConfig proto.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (s, u, v).

    s: A `Tensor`. Has the same type as `a`. Singular values. The values are sorted in reverse order of magnitude, so
      s[..., 0] is the largest value, s[..., 1] is the second largest, etc.
    u: A `Tensor`. Has the same type as `a`. Left singular vectors.
    v: A `Tensor`. Has the same type as `a`. Right singular vectors.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaSvd', name, a, 'max_iter', max_iter, 'epsilon', epsilon, 'precision_config', precision_config)
            _result = _XlaSvdOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_svd((a, max_iter, epsilon, precision_config, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_svd_eager_fallback(a, max_iter=max_iter, epsilon=epsilon, precision_config=precision_config, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_svd, (), dict(a=a, max_iter=max_iter, epsilon=epsilon, precision_config=precision_config, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_svd((a, max_iter, epsilon, precision_config, name), None)
        if _result is not NotImplemented:
            return _result
    max_iter = _execute.make_int(max_iter, 'max_iter')
    epsilon = _execute.make_float(epsilon, 'epsilon')
    precision_config = _execute.make_str(precision_config, 'precision_config')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaSvd', a=a, max_iter=max_iter, epsilon=epsilon, precision_config=precision_config, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_svd, (), dict(a=a, max_iter=max_iter, epsilon=epsilon, precision_config=precision_config, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('max_iter', _op._get_attr_int('max_iter'), 'epsilon', _op.get_attr('epsilon'), 'precision_config', _op.get_attr('precision_config'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaSvd', _inputs_flat, _attrs, _result)
    _result = _XlaSvdOutput._make(_result)
    return _result