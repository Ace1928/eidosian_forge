from ._internal import NDArrayBase
from ..base import _Null
def gamma_like(data=None, alpha=_Null, beta=_Null, out=None, name=None, **kwargs):
    """Draw random samples from a gamma distribution according to the input array shape.

    Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).

    Example::

       gamma(alpha=9, beta=0.5, data=ones(2,2)) = [[ 7.10486984,  3.37695289],
                                                   [ 3.91697288,  3.65933681]]


    Defined in ../src/operator/random/sample_op.cc:L231

    Parameters
    ----------
    alpha : float, optional, default=1
        Alpha parameter (shape) of the gamma distribution.
    beta : float, optional, default=1
        Beta parameter (scale) of the gamma distribution.
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