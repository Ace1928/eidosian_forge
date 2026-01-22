from ._internal import NDArrayBase
from ..base import _Null
def reset_arrays(*data, **kwargs):
    """Set to zero multiple arrays


    Defined in ../src/operator/contrib/reset_arrays.cc:L35

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