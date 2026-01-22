import numpy as np
import scipy.fftpack as fft
from scipy import signal
from statsmodels.tools.validation import array_like, PandasWrapper
def recursive_filter(x, ar_coeff, init=None):
    """
    Autoregressive, or recursive, filtering.

    Parameters
    ----------
    x : array_like
        Time-series data. Should be 1d or n x 1.
    ar_coeff : array_like
        AR coefficients in reverse time order. See Notes for details.
    init : array_like
        Initial values of the time-series prior to the first value of y.
        The default is zero.

    Returns
    -------
    array_like
        Filtered array, number of columns determined by x and ar_coeff. If x
        is a pandas object than a Series is returned.

    Notes
    -----
    Computes the recursive filter ::

        y[n] = ar_coeff[0] * y[n-1] + ...
                + ar_coeff[n_coeff - 1] * y[n - n_coeff] + x[n]

    where n_coeff = len(n_coeff).
    """
    pw = PandasWrapper(x)
    x = array_like(x, 'x')
    ar_coeff = array_like(ar_coeff, 'ar_coeff')
    if init is not None:
        init = array_like(init, 'init')
        if len(init) != len(ar_coeff):
            raise ValueError('ar_coeff must be the same length as init')
    if init is not None:
        zi = signal.lfiltic([1], np.r_[1, -ar_coeff], init, x)
    else:
        zi = None
    y = signal.lfilter([1.0], np.r_[1, -ar_coeff], x, zi=zi)
    if init is not None:
        result = y[0]
    else:
        result = y
    return pw.wrap(result)