from ._internal import NDArrayBase
from ..base import _Null
def linalg_extracttrian(A=None, offset=_Null, lower=_Null, out=None, name=None, **kwargs):
    """Extracts a triangular sub-matrix from a square matrix.
    Input is a tensor *A* of dimension *n >= 2*.

    If *n=2*, then *A* represents a single square matrix from which a triangular sub-matrix is extracted as a 1-dimensional tensor.

    If *n>2*, then *A* represents a batch of square matrices on the trailing two dimensions. The extracted triangular sub-matrices are returned as an *n-1*-dimensional tensor.

    The *offset* and *lower* parameters determine the triangle to be extracted:

    - When *offset = 0* either the lower or upper triangle with respect to the main diagonal is extracted depending on the value of parameter *lower*.
    - When *offset = k > 0* the upper triangle with respect to the k-th diagonal above the main diagonal is extracted. 
    - When *offset = k < 0* the lower triangle with respect to the k-th diagonal below the main diagonal is extracted. 

    .. note:: The operator supports float32 and float64 data types only.

    Examples::

        Single triagonal extraction
        A = [[1.0, 2.0],
             [3.0, 4.0]]

        extracttrian(A) = [1.0, 3.0, 4.0]
        extracttrian(A, lower=False) = [1.0, 2.0, 4.0]
        extracttrian(A, 1) = [2.0]
        extracttrian(A, -1) = [3.0]

        Batch triagonal extraction
        A = [[[1.0, 2.0],
              [3.0, 4.0]],
             [[5.0, 6.0],
              [7.0, 8.0]]]

        extracttrian(A) = [[1.0, 3.0, 4.0],
                           [5.0, 7.0, 8.0]]


    Defined in ../src/operator/tensor/la_op.cc:L604

    Parameters
    ----------
    A : NDArray
        Tensor of square matrices
    offset : int, optional, default='0'
        Offset of the diagonal versus the main diagonal. 0 corresponds to the main diagonal, a negative/positive value to diagonals below/above the main diagonal.
    lower : boolean, optional, default=1
        Refer to the lower triangular matrix if lower=true, refer to the upper otherwise. Only relevant when offset=0

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)