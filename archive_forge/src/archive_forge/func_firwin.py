import math
from cupy.fft import fft, ifft
from cupy.linalg import solve, lstsq, LinAlgError
from cupyx.scipy.linalg import toeplitz, hankel
import cupyx
from cupyx.scipy.signal.windows import get_window
import cupy
import numpy
def firwin(numtaps, cutoff, width=None, window='hamming', pass_zero=True, scale=True, fs=2):
    """
    FIR filter design using the window method.

    This function computes the coefficients of a finite impulse response
    filter.  The filter will have linear phase; it will be Type I if
    `numtaps` is odd and Type II if `numtaps` is even.

    Type II filters always have zero response at the Nyquist frequency, so a
    ValueError exception is raised if firwin is called with `numtaps` even and
    having a passband whose right end is at the Nyquist frequency.

    Parameters
    ----------
    numtaps : int
        Length of the filter (number of coefficients, i.e. the filter
        order + 1).  `numtaps` must be odd if a passband includes the
        Nyquist frequency.
    cutoff : float or 1D array_like
        Cutoff frequency of filter (expressed in the same units as `fs`)
        OR an array of cutoff frequencies (that is, band edges). In the
        latter case, the frequencies in `cutoff` should be positive and
        monotonically increasing between 0 and `fs/2`.  The values 0 and
        `fs/2` must not be included in `cutoff`.
    width : float or None, optional
        If `width` is not None, then assume it is the approximate width
        of the transition region (expressed in the same units as `fs`)
        for use in Kaiser FIR filter design.  In this case, the `window`
        argument is ignored.
    window : string or tuple of string and parameter values, optional
        Desired window to use. See `cusignal.get_window` for a list
        of windows and required parameters.
    pass_zero : {True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop'},
        optional
        If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.
        If False, the DC gain is 0. Can also be a string argument for the
        desired filter type (equivalent to ``btype`` in IIR design functions).
    scale : bool, optional
        Set to True to scale the coefficients so that the frequency
        response is exactly unity at a certain frequency.
        That frequency is either:

        - 0 (DC) if the first passband starts at 0 (i.e. pass_zero
          is True)
        - `fs/2` (the Nyquist frequency) if the first passband ends at
          `fs/2` (i.e the filter is a single band highpass filter);
          center of first passband otherwise
    fs : float, optional
        The sampling frequency of the signal.  Each frequency in `cutoff`
        must be between 0 and ``fs/2``.  Default is 2.

    Returns
    -------
    h : (numtaps,) ndarray
        Coefficients of length `numtaps` FIR filter.

    Raises
    ------
    ValueError
        If any value in `cutoff` is less than or equal to 0 or greater
        than or equal to ``fs/2``, if the values in `cutoff` are not strictly
        monotonically increasing, or if `numtaps` is even but a passband
        includes the Nyquist frequency.

    See Also
    --------
    firwin2
    firls
    minimum_phase
    remez

    Examples
    --------
    Low-pass from 0 to f:

    >>> import cusignal
    >>> numtaps = 3
    >>> f = 0.1
    >>> cusignal.firwin(numtaps, f)
    array([ 0.06799017,  0.86401967,  0.06799017])

    Use a specific window function:

    >>> cusignal.firwin(numtaps, f, window='nuttall')
    array([  3.56607041e-04,   9.99286786e-01,   3.56607041e-04])

    High-pass ('stop' from 0 to f):

    >>> cusignal.firwin(numtaps, f, pass_zero=False)
    array([-0.00859313,  0.98281375, -0.00859313])

    Band-pass:

    >>> f1, f2 = 0.1, 0.2
    >>> cusignal.firwin(numtaps, [f1, f2], pass_zero=False)
    array([ 0.06301614,  0.88770441,  0.06301614])

    Band-stop:

    >>> cusignal.firwin(numtaps, [f1, f2])
    array([-0.00801395,  1.0160279 , -0.00801395])

    Multi-band (passbands are [0, f1], [f2, f3] and [f4, 1]):

    >>> f3, f4 = 0.3, 0.4
    >>> cusignal.firwin(numtaps, [f1, f2, f3, f4])
    array([-0.01376344,  1.02752689, -0.01376344])

    Multi-band (passbands are [f1, f2] and [f3,f4]):

    >>> cusignal.firwin(numtaps, [f1, f2, f3, f4], pass_zero=False)
    array([ 0.04890915,  0.91284326,  0.04890915])

    """
    nyq = 0.5 * fs
    cutoff = cupy.atleast_1d(cutoff) / float(nyq)
    if cutoff.ndim > 1:
        raise ValueError('The cutoff argument must be at most one-dimensional.')
    if cutoff.size == 0:
        raise ValueError('At least one cutoff frequency must be given.')
    if cutoff.min() <= 0 or cutoff.max() >= 1:
        raise ValueError('Invalid cutoff frequency: frequencies must be greater than 0 and less than nyq.')
    if cupy.any(cupy.diff(cutoff) <= 0):
        raise ValueError('Invalid cutoff frequencies: the frequencies must be strictly increasing.')
    if width is not None:
        atten = kaiser_atten(numtaps, float(width) / nyq)
        beta = kaiser_beta(atten)
        window = ('kaiser', beta)
    if isinstance(pass_zero, str):
        if pass_zero in ('bandstop', 'lowpass'):
            if pass_zero == 'lowpass':
                if cutoff.size != 1:
                    raise ValueError('cutoff must have one element if pass_zero=="lowpass", got %s' % (cutoff.shape,))
            elif cutoff.size <= 1:
                raise ValueError('cutoff must have at least two elements if pass_zero=="bandstop", got %s' % (cutoff.shape,))
            pass_zero = True
        elif pass_zero in ('bandpass', 'highpass'):
            if pass_zero == 'highpass':
                if cutoff.size != 1:
                    raise ValueError('cutoff must have one element if pass_zero=="highpass", got %s' % (cutoff.shape,))
            elif cutoff.size <= 1:
                raise ValueError('cutoff must have at least two elements if pass_zero=="bandpass", got %s' % (cutoff.shape,))
            pass_zero = False
        else:
            raise ValueError('pass_zero must be True, False, "bandpass", "lowpass", "highpass", or "bandstop", got {}'.format(pass_zero))
    pass_nyquist = bool(cutoff.size & 1) ^ pass_zero
    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError('A filter with an even number of coefficients must have zero response at the Nyquist rate.')
    cutoff = cupy.hstack(([0.0] * pass_zero, cutoff, [1.0] * pass_nyquist))
    bands = cutoff.reshape(-1, 2)
    win = get_window(window, numtaps, fftbins=False)
    h, hc = _firwin_kernel(win, numtaps, bands, bands.shape[0], scale)
    if scale:
        s = cupy.sum(hc)
        h /= s
        alpha = 0.5 * (numtaps - 1)
        m = cupy.arange(0, numtaps) - alpha
        h = 0
        for left, right in bands:
            h += right * cupy.sinc(right * m)
            h -= left * cupy.sinc(left * m)
        h *= win
        if scale:
            left, right = bands[0]
            if left == 0:
                scale_frequency = 0.0
            elif right == 1:
                scale_frequency = 1.0
            else:
                scale_frequency = 0.5 * (left + right)
            c = cupy.cos(cupy.pi * m * scale_frequency)
            s = cupy.sum(h * c)
            h /= s
    return h