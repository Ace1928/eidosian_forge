import operator
from math import pi
import warnings
import cupy
from cupy.polynomial.polynomial import (
import cupyx.scipy.fft as sp_fft
from cupyx import jit
from cupyx.scipy._lib._util import float_factorial
from cupyx.scipy.signal._polyutils import roots
def group_delay(system, w=512, whole=False, fs=2 * cupy.pi):
    """Compute the group delay of a digital filter.

    The group delay measures by how many samples amplitude envelopes of
    various spectral components of a signal are delayed by a filter.
    It is formally defined as the derivative of continuous (unwrapped) phase::

               d        jw
     D(w) = - -- arg H(e)
              dw

    Parameters
    ----------
    system : tuple of array_like (b, a)
        Numerator and denominator coefficients of a filter transfer function.
    w : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).

        If an array_like, compute the delay at the frequencies given. These
        are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if w is array_like.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

    Returns
    -------
    w : ndarray
        The frequencies at which group delay was computed, in the same units
        as `fs`.  By default, `w` is normalized to the range [0, pi)
        (radians/sample).
    gd : ndarray
        The group delay.

    See Also
    --------
    freqz : Frequency response of a digital filter

    Notes
    -----
    The similar function in MATLAB is called `grpdelay`.

    If the transfer function :math:`H(z)` has zeros or poles on the unit
    circle, the group delay at corresponding frequencies is undefined.
    When such a case arises the warning is raised and the group delay
    is set to 0 at those frequencies.

    For the details of numerical computation of the group delay refer to [1]_.

    References
    ----------
    .. [1] Richard G. Lyons, "Understanding Digital Signal Processing,
           3rd edition", p. 830.

    """
    if w is None:
        w = 512
    if _is_int_type(w):
        if whole:
            w = cupy.linspace(0, 2 * cupy.pi, w, endpoint=False)
        else:
            w = cupy.linspace(0, cupy.pi, w, endpoint=False)
    else:
        w = cupy.atleast_1d(w)
        w = 2 * cupy.pi * w / fs
    b, a = map(cupy.atleast_1d, system)
    c = cupy.convolve(b, a[::-1])
    cr = c * cupy.arange(c.size)
    z = cupy.exp(-1j * w)
    num = cupy.polyval(cr[::-1], z)
    den = cupy.polyval(c[::-1], z)
    gd = cupy.real(num / den) - a.size + 1
    singular = ~cupy.isfinite(gd)
    gd[singular] = 0
    w = w * fs / (2 * cupy.pi)
    return (w, gd)