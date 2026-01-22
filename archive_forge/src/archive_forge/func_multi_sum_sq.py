from ._internal import NDArrayBase
from ..base import _Null
def multi_sum_sq(*data, **kwargs):
    """Compute the sums of squares of multiple arrays


    Defined in ../src/operator/contrib/multi_sum_sq.cc:L35

    Parameters
    ----------
    data : NDArray[]
        Arrays
    num_arrays : int, required
        number of input arrays.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)