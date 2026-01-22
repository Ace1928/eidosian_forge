from ._internal import NDArrayBase
from ..base import _Null
def normal_like(data=None, loc=_Null, scale=_Null, out=None, name=None, **kwargs):
    """Draw random samples from a normal (Gaussian) distribution according to the input array shape.

    Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale*
    (standard deviation).

    Example::

       normal(loc=0, scale=1, data=ones(2,2)) = [[ 1.89171135, -1.16881478],
                                                 [-1.23474145,  1.55807114]]


    Defined in ../src/operator/random/sample_op.cc:L220

    Parameters
    ----------
    loc : float, optional, default=0
        Mean of the distribution.
    scale : float, optional, default=1
        Standard deviation of the distribution.
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