import warnings
import numpy as np
from scipy.linalg import eig
from scipy.special import comb
from scipy.signal import convolve
def daub(p):
    """
    The coefficients for the FIR low-pass filter producing Daubechies wavelets.

    .. deprecated:: 1.12.0

        scipy.signal.daub is deprecated in SciPy 1.12 and will be removed
        in SciPy 1.15. We recommend using PyWavelets instead.

    p>=1 gives the order of the zero at f=1/2.
    There are 2p filter coefficients.

    Parameters
    ----------
    p : int
        Order of the zero at f=1/2, can have values from 1 to 34.

    Returns
    -------
    daub : ndarray
        Return

    """
    warnings.warn(_msg % 'daub', DeprecationWarning, stacklevel=2)
    sqrt = np.sqrt
    if p < 1:
        raise ValueError('p must be at least 1.')
    if p == 1:
        c = 1 / sqrt(2)
        return np.array([c, c])
    elif p == 2:
        f = sqrt(2) / 8
        c = sqrt(3)
        return f * np.array([1 + c, 3 + c, 3 - c, 1 - c])
    elif p == 3:
        tmp = 12 * sqrt(10)
        z1 = 1.5 + sqrt(15 + tmp) / 6 - 1j * (sqrt(15) + sqrt(tmp - 15)) / 6
        z1c = np.conj(z1)
        f = sqrt(2) / 8
        d0 = np.real((1 - z1) * (1 - z1c))
        a0 = np.real(z1 * z1c)
        a1 = 2 * np.real(z1)
        return f / d0 * np.array([a0, 3 * a0 - a1, 3 * a0 - 3 * a1 + 1, a0 - 3 * a1 + 3, 3 - a1, 1])
    elif p < 35:
        if p < 35:
            P = [comb(p - 1 + k, k, exact=1) for k in range(p)][::-1]
            yj = np.roots(P)
        else:
            P = [comb(p - 1 + k, k, exact=1) / 4.0 ** k for k in range(p)][::-1]
            yj = np.roots(P) / 4
        c = np.poly1d([1, 1]) ** p
        q = np.poly1d([1])
        for k in range(p - 1):
            yval = yj[k]
            part = 2 * sqrt(yval * (yval - 1))
            const = 1 - 2 * yval
            z1 = const + part
            if abs(z1) < 1:
                z1 = const - part
            q = q * [1, -z1]
        q = c * np.real(q)
        q = q / np.sum(q) * sqrt(2)
        return q.c[::-1]
    else:
        raise ValueError('Polynomial factorization does not work well for p too large.')