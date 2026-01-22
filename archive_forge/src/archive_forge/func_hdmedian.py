import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def hdmedian(data, axis=-1, var=False):
    """
    Returns the Harrell-Davis estimate of the median along the given axis.

    Parameters
    ----------
    data : ndarray
        Data array.
    axis : int, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.
    var : bool, optional
        Whether to return the variance of the estimate.

    Returns
    -------
    hdmedian : MaskedArray
        The median values.  If ``var=True``, the variance is returned inside
        the masked array.  E.g. for a 1-D array the shape change from (1,) to
        (2,).

    """
    result = hdquantiles(data, [0.5], axis=axis, var=var)
    return result.squeeze()