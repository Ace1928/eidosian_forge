from ._internal import NDArrayBase
from ..base import _Null
def fill_element_0index(lhs=None, mhs=None, rhs=None, out=None, name=None, **kwargs):
    """Fill one element of each line(row for python, column for R/Julia) in lhs according to index indicated by rhs and values indicated by mhs. This function assume rhs uses 0-based index.

    Parameters
    ----------
    lhs : NDArray
        Left operand to the function.
    mhs : NDArray
        Middle operand to the function.
    rhs : NDArray
        Right operand to the function.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)