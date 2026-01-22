import operator
from math import pi
import warnings
import cupy
from cupy.polynomial.polynomial import (
import cupyx.scipy.fft as sp_fft
from cupyx import jit
from cupyx.scipy._lib._util import float_factorial
from cupyx.scipy.signal._polyutils import roots
def gammatone(freq, ftype, order=None, numtaps=None, fs=None):
    """
    Gammatone filter design.

    This function computes the coefficients of an FIR or IIR gammatone
    digital filter [1]_.

    Parameters
    ----------
    freq : float
        Center frequency of the filter (expressed in the same units
        as `fs`).
    ftype : {'fir', 'iir'}
        The type of filter the function generates. If 'fir', the function
        will generate an Nth order FIR gammatone filter. If 'iir', the
        function will generate an 8th order digital IIR filter, modeled as
        as 4th order gammatone filter.
    order : int, optional
        The order of the filter. Only used when ``ftype='fir'``.
        Default is 4 to model the human auditory system. Must be between
        0 and 24.
    numtaps : int, optional
        Length of the filter. Only used when ``ftype='fir'``.
        Default is ``fs*0.015`` if `fs` is greater than 1000,
        15 if `fs` is less than or equal to 1000.
    fs : float, optional
        The sampling frequency of the signal. `freq` must be between
        0 and ``fs/2``. Default is 2.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials of the filter.

    Raises
    ------
    ValueError
        If `freq` is less than or equal to 0 or greater than or equal to
        ``fs/2``, if `ftype` is not 'fir' or 'iir', if `order` is less than
        or equal to 0 or greater than 24 when ``ftype='fir'``

    See Also
    --------
    firwin
    iirfilter

    References
    ----------
    .. [1] Slaney, Malcolm, "An Efficient Implementation of the
        Patterson-Holdsworth Auditory Filter Bank", Apple Computer
        Technical Report 35, 1993, pp.3-8, 34-39.
    """
    freq = float(freq)
    if fs is None:
        fs = 2
    fs = float(fs)
    ftype = ftype.lower()
    filter_types = ['fir', 'iir']
    if not 0 < freq < fs / 2:
        raise ValueError('The frequency must be between 0 and {} (nyquist), but given {}.'.format(fs / 2, freq))
    if ftype not in filter_types:
        raise ValueError('ftype must be either fir or iir.')
    if ftype == 'fir':
        if order is None:
            order = 4
        order = operator.index(order)
        if numtaps is None:
            numtaps = max(int(fs * 0.015), 15)
        numtaps = operator.index(numtaps)
        if not 0 < order <= 24:
            raise ValueError('Invalid order: order must be > 0 and <= 24.')
        t = cupy.arange(numtaps) / fs
        bw = 1.019 * _hz_to_erb(freq)
        b = t ** (order - 1) * cupy.exp(-2 * cupy.pi * bw * t)
        b *= cupy.cos(2 * cupy.pi * freq * t)
        scale_factor = 2 * (2 * cupy.pi * bw) ** order
        scale_factor /= float_factorial(order - 1)
        scale_factor /= fs
        b *= scale_factor
        a = [1.0]
    elif ftype == 'iir':
        if order is not None:
            warnings.warn('order is not used for IIR gammatone filter.')
        if numtaps is not None:
            warnings.warn('numtaps is not used for IIR gammatone filter.')
        b = cupy.empty(5)
        a = cupy.empty(9)
        _gammatone_iir_kernel((9,), (1,), (fs, freq, b, a))
    return (b, a)