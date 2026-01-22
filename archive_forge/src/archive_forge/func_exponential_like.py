from ._internal import NDArrayBase
from ..base import _Null
def exponential_like(data=None, lam=_Null, out=None, name=None, **kwargs):
    """Draw random samples from an exponential distribution according to the input array shape.

    Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).

    Example::

       exponential(lam=4, data=ones(2,2)) = [[ 0.0097189 ,  0.08999364],
                                             [ 0.04146638,  0.31715935]]


    Defined in ../src/operator/random/sample_op.cc:L242

    Parameters
    ----------
    lam : float, optional, default=1
        Lambda parameter (rate) of the exponential distribution.
    data : NDArray
        The input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)