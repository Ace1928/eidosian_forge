import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def cheb2ord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Chebyshev type II filter order selection.

    Return the order of the lowest order digital or analog Chebyshev Type II
    filter that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.)  For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies
        (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    ord : int
        The lowest order for a Chebyshev type II filter that meets specs.
    wn : ndarray or float
        The Chebyshev natural frequency (the "3dB frequency") for use with
        `cheby2` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `cheby2`.

    See Also
    --------
    scipy.signal.cheb2ord
    cheby2 : Filter design using order and critical points
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    """
    _validate_gpass_gstop(gpass, gstop)
    wp, ws, filter_type = _validate_wp_ws(wp, ws, fs, analog)
    passb, stopb = _pre_warp(wp, ws, analog)
    nat, passb = _find_nat_freq(stopb, passb, gpass, gstop, filter_type, 'cheby')
    GSTOP = 10 ** (0.1 * cupy.abs(gstop))
    GPASS = 10 ** (0.1 * cupy.abs(gpass))
    v_pass_stop = cupy.arccosh(cupy.sqrt((GSTOP - 1.0) / (GPASS - 1.0)))
    ord = int(cupy.ceil(v_pass_stop / cupy.arccosh(nat)))
    new_freq = cupy.cosh(1.0 / ord * v_pass_stop)
    new_freq = 1.0 / new_freq
    if filter_type == 1:
        nat = passb / new_freq
    elif filter_type == 2:
        nat = passb * new_freq
    elif filter_type == 3:
        nat = cupy.empty(2, dtype=float)
        nat[0] = new_freq / 2.0 * (passb[0] - passb[1]) + cupy.sqrt(new_freq ** 2 * (passb[1] - passb[0]) ** 2 / 4.0 + passb[1] * passb[0])
        nat[1] = passb[1] * passb[0] / nat[0]
    elif filter_type == 4:
        nat = cupy.empty(2, dtype=float)
        nat[0] = 1.0 / (2.0 * new_freq) * (passb[0] - passb[1]) + cupy.sqrt((passb[1] - passb[0]) ** 2 / (4.0 * new_freq ** 2) + passb[1] * passb[0])
        nat[1] = passb[0] * passb[1] / nat[0]
    wn = _postprocess_wn(nat, analog, fs)
    return (ord, wn)