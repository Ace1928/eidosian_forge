from ._internal import NDArrayBase
from ..base import _Null
def multi_lars(lrs=None, weights_sum_sq=None, grads_sum_sq=None, wds=None, eta=_Null, eps=_Null, rescale_grad=_Null, out=None, name=None, **kwargs):
    """Compute the LARS coefficients of multiple weights and grads from their sums of square"


    Defined in ../src/operator/contrib/multi_lars.cc:L36

    Parameters
    ----------
    lrs : NDArray
        Learning rates to scale by LARS coefficient
    weights_sum_sq : NDArray
        sum of square of weights arrays
    grads_sum_sq : NDArray
        sum of square of gradients arrays
    wds : NDArray
        weight decays
    eta : float, required
        LARS eta
    eps : float, required
        LARS eps
    rescale_grad : float, optional, default=1
        Gradient rescaling factor

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)