from ._internal import NDArrayBase
from ..base import _Null
def extractdiag(A=None, offset=_Null, out=None, name=None, **kwargs):
    """Extracts the diagonal entries of a square matrix.
    Input is a tensor *A* of dimension *n >= 2*.

    If *n=2*, then *A* represents a single square matrix which diagonal elements get extracted as a 1-dimensional tensor.

    If *n>2*, then *A* represents a batch of square matrices on the trailing two dimensions. The extracted diagonals are returned as an *n-1*-dimensional tensor.

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

        Single matrix diagonal extraction
        A = [[1.0, 2.0],
             [3.0, 4.0]]

        extractdiag(A) = [1.0, 4.0]

        extractdiag(A, 1) = [2.0]

        Batch matrix diagonal extraction
        A = [[[1.0, 2.0],
              [3.0, 4.0]],
             [[5.0, 6.0],
              [7.0, 8.0]]]

        extractdiag(A) = [[1.0, 4.0],
                          [5.0, 8.0]]


    Defined in ../src/operator/tensor/la_op.cc:L494

    Parameters
    ----------
    A : NDArray
        Tensor of square matrices
    offset : int, optional, default='0'
        Offset of the diagonal versus the main diagonal. 0 corresponds to the main diagonal, a negative/positive value to diagonals below/above the main diagonal.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)