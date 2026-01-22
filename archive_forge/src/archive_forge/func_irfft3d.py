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
def irfft3d(input: _atypes.TensorFuzzingAnnotation[TV_IRFFT3D_Tcomplex], fft_length: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Treal: TV_IRFFT3D_Treal=_dtypes.float32, name=None) -> _atypes.TensorFuzzingAnnotation[TV_IRFFT3D_Treal]:
    """Inverse 3D real-valued fast Fourier transform.

  Computes the inverse 3-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most 3 dimensions of `input`.

  The inner-most 3 dimensions of `input` are assumed to be the result of `RFFT3D`:
  The inner-most dimension contains the `fft_length / 2 + 1` unique components of
  the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
  from the size of the inner-most 3 dimensions of `input`. If the FFT length used
  to compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Along each axis `IRFFT3D` is computed on, if `fft_length` (or
  `fft_length / 2 + 1` for the inner-most dimension) is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [3]. The FFT length for each dimension.
    Treal: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Treal`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'IRFFT3D', name, input, fft_length, 'Treal', Treal)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return irfft3d_eager_fallback(input, fft_length, Treal=Treal, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if Treal is None:
        Treal = _dtypes.float32
    Treal = _execute.make_type(Treal, 'Treal')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('IRFFT3D', input=input, fft_length=fft_length, Treal=Treal, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Treal', _op._get_attr_type('Treal'), 'Tcomplex', _op._get_attr_type('Tcomplex'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('IRFFT3D', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result