import numpy as np
from scipy import signal
from numpy.testing import assert_array_equal, assert_array_almost_equal
def movvar(x, windowsize=3, lag='lagged'):
    """moving window variance


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
        moving variance, with same shape as x


    """
    m1 = movmoment(x, 1, windowsize=windowsize, lag=lag)
    m2 = movmoment(x, 2, windowsize=windowsize, lag=lag)
    return m2 - m1 * m1