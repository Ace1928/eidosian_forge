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
@tf_export('linalg.qr', v1=['linalg.qr', 'qr'])
@deprecated_endpoints('qr')
def qr(input: _atypes.TensorFuzzingAnnotation[TV_Qr_T], full_matrices: bool=False, name=None):
    """Computes the QR decompositions of one or more matrices.

  Computes the QR decomposition of each inner matrix in `tensor` such that
  `tensor[..., :, :] = q[..., :, :] * r[..., :,:])`

  Currently, the gradient for the QR decomposition is well-defined only when
  the first `P` columns of the inner matrix are linearly independent, where
  `P` is the minimum of `M` and `N`, the 2 inner-most dimmensions of `tensor`.

  ```python
  # a is a tensor.
  # q is a tensor of orthonormal matrices.
  # r is a tensor of upper triangular matrices.
  q, r = qr(a)
  q_full, r_full = qr(a, full_matrices=True)
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`, `half`, `complex64`, `complex128`.
      A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
      form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.
    full_matrices: An optional `bool`. Defaults to `False`.
      If true, compute full-sized `q` and `r`. If false
      (the default), compute only the leading `P` columns of `q`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (q, r).

    q: A `Tensor`. Has the same type as `input`.
    r: A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Qr', name, input, 'full_matrices', full_matrices)
            _result = _QrOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_qr((input, full_matrices, name), None)
            if _result is not NotImplemented:
                return _result
            return qr_eager_fallback(input, full_matrices=full_matrices, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(qr, (), dict(input=input, full_matrices=full_matrices, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_qr((input, full_matrices, name), None)
        if _result is not NotImplemented:
            return _result
    if full_matrices is None:
        full_matrices = False
    full_matrices = _execute.make_bool(full_matrices, 'full_matrices')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('Qr', input=input, full_matrices=full_matrices, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(qr, (), dict(input=input, full_matrices=full_matrices, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('full_matrices', _op._get_attr_bool('full_matrices'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Qr', _inputs_flat, _attrs, _result)
    _result = _QrOutput._make(_result)
    return _result