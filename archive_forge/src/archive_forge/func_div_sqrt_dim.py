from ._internal import NDArrayBase
from ..base import _Null
def div_sqrt_dim(data=None, out=None, name=None, **kwargs):
    """Rescale the input by the square root of the channel dimension.

       out = data / sqrt(data.shape[-1])



    Defined in ../src/operator/contrib/transformer.cc:L832

    Parameters
    ----------
    data : NDArray
        The input array.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)