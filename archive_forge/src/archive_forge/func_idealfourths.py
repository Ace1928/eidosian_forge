import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def idealfourths(data, axis=None):
    """
    Returns an estimate of the lower and upper quartiles.

    Uses the ideal fourths algorithm.

    Parameters
    ----------
    data : array_like
        Input array.
    axis : int, optional
        Axis along which the quartiles are estimated. If None, the arrays are
        flattened.

    Returns
    -------
    idealfourths : {list of floats, masked array}
        Returns the two internal values that divide `data` into four parts
        using the ideal fourths algorithm either along the flattened array
        (if `axis` is None) or along `axis` of `data`.

    """

    def _idf(data):
        x = data.compressed()
        n = len(x)
        if n < 3:
            return [np.nan, np.nan]
        j, h = divmod(n / 4.0 + 5 / 12.0, 1)
        j = int(j)
        qlo = (1 - h) * x[j - 1] + h * x[j]
        k = n - j
        qup = (1 - h) * x[k] + h * x[k - 1]
        return [qlo, qup]
    data = ma.sort(data, axis=axis).view(MaskedArray)
    if axis is None:
        return _idf(data)
    else:
        return ma.apply_along_axis(_idf, axis, data)