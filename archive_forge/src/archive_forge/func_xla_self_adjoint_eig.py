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
@tf_export('xla_self_adjoint_eig')
def xla_self_adjoint_eig(a: _atypes.TensorFuzzingAnnotation[TV_XlaSelfAdjointEig_T], lower: bool, max_iter: int, epsilon: float, name=None):
    """Computes the eigen decomposition of a batch of self-adjoint matrices

  (Note: Only real inputs are supported).

  Computes the eigenvalues and eigenvectors of the innermost N-by-N matrices in
  tensor such that tensor[...,:,:] * v[..., :,i] = e[..., i] * v[...,:,i], for
  i=0...N-1.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      the input tensor.
    lower: A `bool`.
      a boolean specifies whether the calculation is done with the lower
      triangular part or the upper triangular part.
    max_iter: An `int`.
      maximum number of sweep update, i.e., the whole lower triangular
      part or upper triangular part based on parameter lower. Heuristically, it has
      been argued that approximately logN sweeps are needed in practice (Ref: Golub &
      van Loan "Matrix Computation").
    epsilon: A `float`. the tolerance ratio.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (w, v).

    w: A `Tensor`. Has the same type as `a`. The eigenvalues in ascending order, each repeated according to its
      multiplicity.
    v: A `Tensor`. Has the same type as `a`. The column v[..., :, i] is the normalized eigenvector corresponding to the
      eigenvalue w[..., i].
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaSelfAdjointEig', name, a, 'lower', lower, 'max_iter', max_iter, 'epsilon', epsilon)
            _result = _XlaSelfAdjointEigOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_self_adjoint_eig((a, lower, max_iter, epsilon, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_self_adjoint_eig_eager_fallback(a, lower=lower, max_iter=max_iter, epsilon=epsilon, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_self_adjoint_eig, (), dict(a=a, lower=lower, max_iter=max_iter, epsilon=epsilon, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_self_adjoint_eig((a, lower, max_iter, epsilon, name), None)
        if _result is not NotImplemented:
            return _result
    lower = _execute.make_bool(lower, 'lower')
    max_iter = _execute.make_int(max_iter, 'max_iter')
    epsilon = _execute.make_float(epsilon, 'epsilon')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaSelfAdjointEig', a=a, lower=lower, max_iter=max_iter, epsilon=epsilon, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_self_adjoint_eig, (), dict(a=a, lower=lower, max_iter=max_iter, epsilon=epsilon, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('lower', _op._get_attr_bool('lower'), 'max_iter', _op._get_attr_int('max_iter'), 'epsilon', _op.get_attr('epsilon'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaSelfAdjointEig', _inputs_flat, _attrs, _result)
    _result = _XlaSelfAdjointEigOutput._make(_result)
    return _result