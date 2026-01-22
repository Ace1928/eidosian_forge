import math
from cupy.fft import fft, ifft
from cupy.linalg import solve, lstsq, LinAlgError
from cupyx.scipy.linalg import toeplitz, hankel
import cupyx
from cupyx.scipy.signal.windows import get_window
import cupy
import numpy
def kaiserord(ripple, width):
    """
    Determine the filter window parameters for the Kaiser window method.

    The parameters returned by this function are generally used to create
    a finite impulse response filter using the window method, with either
    `firwin` or `firwin2`.

    Parameters
    ----------
    ripple : float
        Upper bound for the deviation (in dB) of the magnitude of the
        filter's frequency response from that of the desired filter (not
        including frequencies in any transition intervals). That is, if w
        is the frequency expressed as a fraction of the Nyquist frequency,
        A(w) is the actual frequency response of the filter and D(w) is the
        desired frequency response, the design requirement is that::

            abs(A(w) - D(w))) < 10**(-ripple/20)

        for 0 <= w <= 1 and w not in a transition interval.
    width : float
        Width of transition region, normalized so that 1 corresponds to pi
        radians / sample. That is, the frequency is expressed as a fraction
        of the Nyquist frequency.

    Returns
    -------
    numtaps : int
        The length of the Kaiser window.
    beta : float
        The beta parameter for the Kaiser window.

    See Also
    --------
    scipy.signal.kaiserord


    """
    A = abs(ripple)
    if A < 8:
        raise ValueError('Requested maximum ripple attentuation %f is too small for the Kaiser formula.' % A)
    beta = kaiser_beta(A)
    numtaps = (A - 7.95) / 2.285 / (cupy.pi * width) + 1
    return (int(numpy.ceil(numtaps)), beta)