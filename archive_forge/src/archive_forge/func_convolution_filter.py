import numpy as np
import scipy.fftpack as fft
from scipy import signal
from statsmodels.tools.validation import array_like, PandasWrapper
def convolution_filter(x, filt, nsides=2):
    """
    Linear filtering via convolution. Centered and backward displaced moving
    weighted average.

    Parameters
    ----------
    x : array_like
        data array, 1d or 2d, if 2d then observations in rows
    filt : array_like
        Linear filter coefficients in reverse time-order. Should have the
        same number of dimensions as x though if 1d and ``x`` is 2d will be
        coerced to 2d.
    nsides : int, optional
        If 2, a centered moving average is computed using the filter
        coefficients. If 1, the filter coefficients are for past values only.
        Both methods use scipy.signal.convolve.

    Returns
    -------
    y : ndarray, 2d
        Filtered array, number of columns determined by x and filt. If a
        pandas object is given, a pandas object is returned. The index of
        the return is the exact same as the time period in ``x``

    Notes
    -----
    In nsides == 1, x is filtered ::

        y[n] = filt[0]*x[n-1] + ... + filt[n_filt-1]*x[n-n_filt]

    where n_filt is len(filt).

    If nsides == 2, x is filtered around lag 0 ::

        y[n] = filt[0]*x[n - n_filt/2] + ... + filt[n_filt / 2] * x[n]
               + ... + x[n + n_filt/2]

    where n_filt is len(filt). If n_filt is even, then more of the filter
    is forward in time than backward.

    If filt is 1d or (nlags,1) one lag polynomial is applied to all
    variables (columns of x). If filt is 2d, (nlags, nvars) each series is
    independently filtered with its own lag polynomial, uses loop over nvar.
    This is different than the usual 2d vs 2d convolution.

    Filtering is done with scipy.signal.convolve, so it will be reasonably
    fast for medium sized data. For large data fft convolution would be
    faster.
    """
    if nsides == 1:
        trim_head = len(filt) - 1
        trim_tail = None
    elif nsides == 2:
        trim_head = int(np.ceil(len(filt) / 2.0) - 1) or None
        trim_tail = int(np.ceil(len(filt) / 2.0) - len(filt) % 2) or None
    else:
        raise ValueError('nsides must be 1 or 2')
    pw = PandasWrapper(x)
    x = array_like(x, 'x', maxdim=2)
    filt = array_like(filt, 'filt', ndim=x.ndim)
    if filt.ndim == 1 or min(filt.shape) == 1:
        result = signal.convolve(x, filt, mode='valid')
    else:
        nlags = filt.shape[0]
        nvar = x.shape[1]
        result = np.zeros((x.shape[0] - nlags + 1, nvar))
        if nsides == 2:
            for i in range(nvar):
                result[:, i] = signal.convolve(x[:, i], filt[:, i], mode='valid')
        elif nsides == 1:
            for i in range(nvar):
                result[:, i] = signal.convolve(x[:, i], np.r_[0, filt[:, i]], mode='valid')
    result = _pad_nans(result, trim_head, trim_tail)
    return pw.wrap(result)