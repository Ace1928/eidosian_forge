from ._internal import NDArrayBase
from ..base import _Null
def negative_binomial_like(data=None, k=_Null, p=_Null, out=None, name=None, **kwargs):
    """Draw random samples from a negative binomial distribution according to the input array shape.

    Samples are distributed according to a negative binomial distribution parametrized by
    *k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).
    Samples will always be returned as a floating point data type.

    Example::

       negative_binomial(k=3, p=0.4, data=ones(2,2)) = [[ 4.,  7.],
                                                        [ 2.,  5.]]


    Defined in ../src/operator/random/sample_op.cc:L267

    Parameters
    ----------
    k : int, optional, default='1'
        Limit of unsuccessful experiments.
    p : float, optional, default=1
        Failure probability in each experiment.
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