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
Initialize an `LinearOperatorCirculant`.

    This `LinearOperator` is initialized to have shape `[B1,...,Bb, N, N]`
    by providing `spectrum`, a `[B1,...,Bb, N0, N1, N2]` `Tensor`
    with `N0*N1*N2 = N`.

    If `input_output_dtype = DTYPE`:

    * Arguments to methods such as `matmul` or `solve` must be `DTYPE`.
    * Values returned by all methods, such as `matmul` or `determinant` will be
      cast to `DTYPE`.

    Note that if the spectrum is not Hermitian, then this operator corresponds
    to a complex matrix with non-zero imaginary part.  In this case, setting
    `input_output_dtype` to a real type will forcibly cast the output to be
    real, resulting in incorrect results!

    If on the other hand the spectrum is Hermitian, then this operator
    corresponds to a real-valued matrix, and setting `input_output_dtype` to
    a real type is fine.

    Args:
      spectrum:  Shape `[B1,...,Bb, N0, N1, N2]` `Tensor`.  Allowed dtypes:
        `float16`, `float32`, `float64`, `complex64`, `complex128`.
        Type can be different than `input_output_dtype`
      input_output_dtype: `dtype` for input/output.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `spectrum` is real, this will always be true.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the real part of all eigenvalues is positive.  We do not require
        the operator to be self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix
            #Extension_for_non_symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name:  A name to prepend to all ops created by this class.
    