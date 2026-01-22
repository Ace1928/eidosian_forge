import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupy._core.internal import _normalize_axis_index
from cupyx.scipy.signal._signaltools import lfilter
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal._iir_utils import collapse_2d, apply_iir_sos
def symiirorder1(input, c0, z1, precision=-1.0):
    """
    Implement a smoothing IIR filter with mirror-symmetric boundary conditions
    using a cascade of first-order sections.  The second section uses a
    reversed sequence.  This implements a system with the following
    transfer function and mirror-symmetric boundary conditions::

                           c0
           H(z) = ---------------------
                   (1-z1/z) (1 - z1 z)

    The resulting signal will have mirror symmetric boundary conditions
    as well.

    Parameters
    ----------
    input : ndarray
        The input signal.
    c0, z1 : scalar
        Parameters in the transfer function.
    precision :
        Specifies the precision for calculating initial conditions
        of the recursive filter based on mirror-symmetric input.

    Returns
    -------
    output : ndarray
        The filtered signal.
    """
    c0 = cupy.asarray([c0], input.dtype)
    z1 = cupy.asarray([z1], input.dtype)
    if cupy.abs(z1) >= 1:
        raise ValueError('|z1| must be less than 1.0')
    if precision <= 0.0 or precision > 1.0:
        precision = cupy.finfo(input.dtype).resolution
    precision *= precision
    pos = cupy.arange(1, input.size + 1, dtype=input.dtype)
    pow_z1 = z1 ** pos
    diff = pow_z1 * cupy.conjugate(pow_z1)
    cum_poly = cupy.cumsum(pow_z1 * input) + input[0]
    all_valid = diff <= precision
    zi = _find_initial_cond(all_valid, cum_poly, input.size)
    if cupy.isnan(zi):
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    a = cupy.r_[1, -z1]
    a = a.astype(input.dtype)
    y1, _ = lfilter(cupy.ones(1, dtype=input.dtype), a, input[1:], zi=zi)
    y1 = cupy.r_[zi, y1]
    zi = -c0 / (z1 - 1.0) * y1[-1]
    a = cupy.r_[1, -z1]
    a = a.astype(input.dtype)
    out, _ = lfilter(c0, a, y1[:-1][::-1], zi=zi)
    return cupy.r_[out[::-1], zi]