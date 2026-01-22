from ._internal import NDArrayBase
from ..base import _Null
def linalg_slogdet(A=None, out=None, name=None, **kwargs):
    """Compute the sign and log of the determinant of a matrix.
    Input is a tensor *A* of dimension *n >= 2*.

    If *n=2*, *A* is a square matrix. We compute:

      *sign* = *sign(det(A))*
      *logabsdet* = *log(abs(det(A)))*

    If *n>2*, *slogdet* is performed separately on the trailing two dimensions
    for all inputs (batch mode).

    .. note:: The operator supports float32 and float64 data types only.
    .. note:: The gradient is not properly defined on sign, so the gradient of
              it is not backwarded.
    .. note:: No gradient is backwarded when A is non-invertible. Please see
              the docs of operator det for detail.

    Examples::

       Single matrix signed log determinant
       A = [[2., 3.], [1., 4.]]
       sign, logabsdet = slogdet(A)
       sign = [1.]
       logabsdet = [1.609438]

       Batch matrix signed log determinant
       A = [[[2., 3.], [1., 4.]],
            [[1., 2.], [2., 4.]],
            [[1., 2.], [4., 3.]]]
       sign, logabsdet = slogdet(A)
       sign = [1., 0., -1.]
       logabsdet = [1.609438, -inf, 1.609438]


    Defined in ../src/operator/tensor/la_op.cc:L1033

    Parameters
    ----------
    A : NDArray
        Tensor of square matrix

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)