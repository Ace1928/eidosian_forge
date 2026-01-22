import numpy as np
from statsmodels.tools.validation import array_like
def stde(x1, x2, ddof=0, axis=0):
    """standard deviation of error

    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated

    Returns
    -------
    stde : ndarray or float
       standard deviation of difference along given axis.

    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.std(x1 - x2, ddof=ddof, axis=axis)