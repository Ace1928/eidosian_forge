import numpy as np
from statsmodels.tools.validation import array_like
def rmspe(y, y_hat, axis=0, zeros=np.nan):
    """
    Root Mean Squared Percentage Error

    Parameters
    ----------
    y : array_like
      The actual value.
    y_hat : array_like
       The predicted value.
    axis : int
       Axis along which the summary statistic is calculated
    zeros : float
       Value to assign to error where y is zero

    Returns
    -------
    rmspe : ndarray or float
       Root Mean Squared Percentage Error along given axis.
    """
    y_hat = np.asarray(y_hat)
    y = np.asarray(y)
    error = y - y_hat
    loc = y != 0
    loc = loc.ravel()
    percentage_error = np.full_like(error, zeros)
    percentage_error.flat[loc] = error.flat[loc] / y.flat[loc]
    mspe = np.nanmean(percentage_error ** 2, axis=axis) * 100
    return np.sqrt(mspe)