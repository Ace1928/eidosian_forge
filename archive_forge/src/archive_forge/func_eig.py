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
def eig(input: _atypes.TensorFuzzingAnnotation[TV_Eig_T], Tout: TV_Eig_Tout, compute_v: bool=True, name=None):
    """Computes the eigen decomposition of one or more square matrices.

  Computes the eigenvalues and (optionally) right eigenvectors of each inner matrix in
  `input` such that `input[..., :, :] = v[..., :, :] * diag(e[..., :])`. The eigenvalues
  are sorted in non-decreasing order.

  ```python
  # a is a tensor.
  # e is a tensor of eigenvalues.
  # v is a tensor of eigenvectors.
  e, v = eig(a)
  e = eig(a, compute_v=False)
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `complex64`, `complex128`.
      `Tensor` input of shape `[N, N]`.
    Tout: A `tf.DType` from: `tf.complex64, tf.complex128`.
    compute_v: An optional `bool`. Defaults to `True`.
      If `True` then eigenvectors will be computed and returned in `v`.
      Otherwise, only the eigenvalues will be computed.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (e, v).

    e: A `Tensor` of type `Tout`.
    v: A `Tensor` of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Eig', name, input, 'compute_v', compute_v, 'Tout', Tout)
            _result = _EigOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return eig_eager_fallback(input, compute_v=compute_v, Tout=Tout, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    Tout = _execute.make_type(Tout, 'Tout')
    if compute_v is None:
        compute_v = True
    compute_v = _execute.make_bool(compute_v, 'compute_v')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Eig', input=input, Tout=Tout, compute_v=compute_v, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('compute_v', _op._get_attr_bool('compute_v'), 'T', _op._get_attr_type('T'), 'Tout', _op._get_attr_type('Tout'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Eig', _inputs_flat, _attrs, _result)
    _result = _EigOutput._make(_result)
    return _result