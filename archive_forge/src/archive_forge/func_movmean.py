import numpy as np
from scipy import signal
from numpy.testing import assert_array_equal, assert_array_almost_equal
def movmean(x, windowsize=3, lag='lagged'):
    """moving window mean


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
        moving mean, with same shape as x


    Notes
    -----
    for leading and lagging the data array x is extended by the closest value of the array


    """
    return movmoment(x, 1, windowsize=windowsize, lag=lag)