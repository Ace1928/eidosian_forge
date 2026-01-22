import numpy as np
from scipy import signal
from numpy.testing import assert_array_equal, assert_array_almost_equal
def movmoment(x, k, windowsize=3, lag='lagged'):
    """non-central moment


    Parameters
    ----------
    x : ndarray
       time series data
    windsize : int
       window size
    lag : 'lagged', 'centered', or 'leading'
       location of window relative to current position

    Returns
    -------
    mk : ndarray
        k-th moving non-central moment, with same shape as x


    Notes
    -----
    If data x is 2d, then moving moment is calculated for each
    column.

    """
    windsize = windowsize
    if lag == 'lagged':
        lead = -0
        sl = slice(windsize - 1 or None, -2 * (windsize - 1) or None)
    elif lag == 'centered':
        lead = -windsize // 2
        sl = slice(windsize - 1 + windsize // 2 or None, -(windsize - 1) - windsize // 2 or None)
    elif lag == 'leading':
        lead = -windsize + 2
        sl = slice(2 * (windsize - 1) + 1 + lead or None, -(2 * (windsize - 1) + lead) + 1 or None)
    else:
        raise ValueError
    avgkern = np.ones(windowsize) / float(windowsize)
    xext = expandarr(x, windsize - 1)
    print(sl)
    if xext.ndim == 1:
        return np.correlate(xext ** k, avgkern, 'full')[sl]
    else:
        print(xext.shape)
        print(avgkern[:, None].shape)
        return signal.correlate(xext ** k, avgkern[:, None], 'full')[sl, :]