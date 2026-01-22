from math import pi
import math
import cupy
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._iir_filter_conversions import (
def iirdesign(wp, ws, gpass, gstop, analog=False, ftype='ellip', output='ba', fs=None):
    """Complete IIR digital and analog filter design.

    Given passband and stopband frequencies and gains, construct an analog or
    digital IIR filter of minimum order for a given basic type. Return the
    output in numerator, denominator ('ba'), pole-zero ('zpk') or second order
    sections ('sos') form.

    Parameters
    ----------
    wp, ws : float or array like, shape (2,)
        Passband and stopband edge frequencies. Possible values are scalars
        (for lowpass and highpass filters) or ranges (for bandpass and bandstop
        filters).
        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies
        (e.g., rad/s). Note, that for bandpass and bandstop filters passband
        must lie strictly inside stopband or vice versa.
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:

            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'

    output : {'ba', 'zpk', 'sos'}, optional
        Filter form of the output:

            - second-order sections (recommended): 'sos'
            - numerator/denominator (default)    : 'ba'
            - pole-zero                          : 'zpk'

        In general the second-order sections ('sos') form  is
        recommended because inferring the coefficients for the
        numerator/denominator form ('ba') suffers from numerical
        instabilities. For reasons of backward compatibility the default
        form is the numerator/denominator form ('ba'), where the 'b'
        and the 'a' in 'ba' refer to the commonly used names of the
        coefficients used.

        Note: Using the second-order sections form ('sos') is sometimes
        associated with additional computational costs: for
        data-intense use cases it is therefore recommended to also
        investigate the numerator/denominator form ('ba').

    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output='sos'``.

    See Also
    --------
    scipy.signal.iirdesign
    butter : Filter design using order and critical points
    cheby1, cheby2, ellip, bessel
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    """
    try:
        ordfunc = filter_dict[ftype][1]
    except KeyError as e:
        raise ValueError('Invalid IIR filter type: %s' % ftype) from e
    except IndexError as e:
        raise ValueError('%s does not have order selection. Use iirfilter function.' % ftype) from e
    _validate_gpass_gstop(gpass, gstop)
    wp = cupy.atleast_1d(wp)
    ws = cupy.atleast_1d(ws)
    if wp.shape[0] != ws.shape[0] or wp.shape not in [(1,), (2,)]:
        raise ValueError('wp and ws must have one or two elements each, andthe same shape, got %s and %s' % (wp.shape, ws.shape))
    if any(wp <= 0) or any(ws <= 0):
        raise ValueError('Values for wp, ws must be greater than 0')
    if not analog:
        if fs is None:
            if any(wp >= 1) or any(ws >= 1):
                raise ValueError('Values for wp, ws must be less than 1')
        elif any(wp >= fs / 2) or any(ws >= fs / 2):
            raise ValueError('Values for wp, ws must be less than fs/2 (fs={} -> fs/2={})'.format(fs, fs / 2))
    if wp.shape[0] == 2:
        if not (ws[0] < wp[0] and wp[1] < ws[1] or (wp[0] < ws[0] and ws[1] < wp[1])):
            raise ValueError('Passband must lie strictly inside stopband or vice versa')
    band_type = 2 * (len(wp) - 1)
    band_type += 1
    if wp[0] >= ws[0]:
        band_type += 1
    btype = {1: 'lowpass', 2: 'highpass', 3: 'bandstop', 4: 'bandpass'}[band_type]
    N, Wn = ordfunc(wp, ws, gpass, gstop, analog=analog, fs=fs)
    return iirfilter(N, Wn, rp=gpass, rs=gstop, analog=analog, btype=btype, ftype=ftype, output=output, fs=fs)