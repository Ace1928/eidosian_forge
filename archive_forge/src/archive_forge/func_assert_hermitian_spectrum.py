import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export
def assert_hermitian_spectrum(self, name='assert_hermitian_spectrum'):
    """Returns an `Op` that asserts this operator has Hermitian spectrum.

    This operator corresponds to a real-valued matrix if and only if its
    spectrum is Hermitian.

    Args:
      name:  A name to give this `Op`.

    Returns:
      An `Op` that asserts this operator has Hermitian spectrum.
    """
    eps = np.finfo(self.dtype.real_dtype.as_numpy_dtype).eps
    with self._name_scope(name):
        max_err = eps * self.domain_dimension_tensor()
        imag_convolution_kernel = math_ops.imag(self.convolution_kernel())
        return check_ops.assert_less(math_ops.abs(imag_convolution_kernel), max_err, message='Spectrum was not Hermitian')