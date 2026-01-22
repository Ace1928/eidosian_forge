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
def rfft2d(input: _atypes.TensorFuzzingAnnotation[TV_RFFT2D_Treal], fft_length: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Tcomplex: TV_RFFT2D_Tcomplex=_dtypes.complex64, name=None) -> _atypes.TensorFuzzingAnnotation[TV_RFFT2D_Tcomplex]:
    """2D real-valued fast Fourier transform.

  Computes the 2-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most 2 dimensions of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
  `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
  of `output`: the zero-frequency term, followed by the `fft_length / 2`
  positive-frequency terms.

  Along each axis `RFFT2D` is computed on, if `fft_length` is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [2]. The FFT length for each dimension.
    Tcomplex: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tcomplex`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RFFT2D', name, input, fft_length, 'Tcomplex', Tcomplex)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return rfft2d_eager_fallback(input, fft_length, Tcomplex=Tcomplex, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if Tcomplex is None:
        Tcomplex = _dtypes.complex64
    Tcomplex = _execute.make_type(Tcomplex, 'Tcomplex')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RFFT2D', input=input, fft_length=fft_length, Tcomplex=Tcomplex, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Treal', _op._get_attr_type('Treal'), 'Tcomplex', _op._get_attr_type('Tcomplex'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RFFT2D', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result