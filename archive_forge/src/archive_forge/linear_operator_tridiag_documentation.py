from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
Initialize a `LinearOperatorTridiag`.

    Args:
      diagonals: `Tensor` or list of `Tensor`s depending on `diagonals_format`.

        If `diagonals_format=sequence`, this is a list of three `Tensor`'s each
        with shape `[B1, ..., Bb, N]`, `b >= 0, N >= 0`, representing the
        superdiagonal, diagonal and subdiagonal in that order. Note the
        superdiagonal is padded with an element in the last position, and the
        subdiagonal is padded with an element in the front.

        If `diagonals_format=matrix` this is a `[B1, ... Bb, N, N]` shaped
        `Tensor` representing the full tridiagonal matrix.

        If `diagonals_format=compact` this is a `[B1, ... Bb, 3, N]` shaped
        `Tensor` with the second to last dimension indexing the
        superdiagonal, diagonal and subdiagonal in that order. Note the
        superdiagonal is padded with an element in the last position, and the
        subdiagonal is padded with an element in the front.

        In every case, these `Tensor`s are all floating dtype.
      diagonals_format: one of `matrix`, `sequence`, or `compact`. Default is
        `compact`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `diag.dtype` is real, this is auto-set to `True`.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.

    Raises:
      TypeError:  If `diag.dtype` is not an allowed type.
      ValueError:  If `diag.dtype` is real, and `is_self_adjoint` is not `True`.
    