import warnings
import numpy as np
from scipy.linalg import eig
from scipy.special import comb
from scipy.signal import convolve
def cascade(hk, J=7):
    """
    Return (x, phi, psi) at dyadic points ``K/2**J`` from filter coefficients.

    .. deprecated:: 1.12.0

        scipy.signal.cascade is deprecated in SciPy 1.12 and will be removed
        in SciPy 1.15. We recommend using PyWavelets instead.

    Parameters
    ----------
    hk : array_like
        Coefficients of low-pass filter.
    J : int, optional
        Values will be computed at grid points ``K/2**J``. Default is 7.

    Returns
    -------
    x : ndarray
        The dyadic points ``K/2**J`` for ``K=0...N * (2**J)-1`` where
        ``len(hk) = len(gk) = N+1``.
    phi : ndarray
        The scaling function ``phi(x)`` at `x`:
        ``phi(x) = sum(hk * phi(2x-k))``, where k is from 0 to N.
    psi : ndarray, optional
        The wavelet function ``psi(x)`` at `x`:
        ``phi(x) = sum(gk * phi(2x-k))``, where k is from 0 to N.
        `psi` is only returned if `gk` is not None.

    Notes
    -----
    The algorithm uses the vector cascade algorithm described by Strang and
    Nguyen in "Wavelets and Filter Banks".  It builds a dictionary of values
    and slices for quick reuse.  Then inserts vectors into final vector at the
    end.

    """
    warnings.warn(_msg % 'cascade', DeprecationWarning, stacklevel=2)
    N = len(hk) - 1
    if J > 30 - np.log2(N + 1):
        raise ValueError('Too many levels.')
    if J < 1:
        raise ValueError('Too few levels.')
    nn, kk = np.ogrid[:N, :N]
    s2 = np.sqrt(2)
    thk = np.r_[hk, 0]
    gk = qmf(hk)
    tgk = np.r_[gk, 0]
    indx1 = np.clip(2 * nn - kk, -1, N + 1)
    indx2 = np.clip(2 * nn - kk + 1, -1, N + 1)
    m = np.empty((2, 2, N, N), 'd')
    m[0, 0] = np.take(thk, indx1, 0)
    m[0, 1] = np.take(thk, indx2, 0)
    m[1, 0] = np.take(tgk, indx1, 0)
    m[1, 1] = np.take(tgk, indx2, 0)
    m *= s2
    x = np.arange(0, N * (1 << J), dtype=float) / (1 << J)
    phi = 0 * x
    psi = 0 * x
    lam, v = eig(m[0, 0])
    ind = np.argmin(np.absolute(lam - 1))
    v = np.real(v[:, ind])
    sm = np.sum(v)
    if sm < 0:
        v = -v
        sm = -sm
    bitdic = {'0': v / sm}
    bitdic['1'] = np.dot(m[0, 1], bitdic['0'])
    step = 1 << J
    phi[::step] = bitdic['0']
    phi[1 << J - 1::step] = bitdic['1']
    psi[::step] = np.dot(m[1, 0], bitdic['0'])
    psi[1 << J - 1::step] = np.dot(m[1, 1], bitdic['0'])
    prevkeys = ['1']
    for level in range(2, J + 1):
        newkeys = ['%d%s' % (xx, yy) for xx in [0, 1] for yy in prevkeys]
        fac = 1 << J - level
        for key in newkeys:
            num = 0
            for pos in range(level):
                if key[pos] == '1':
                    num += 1 << level - 1 - pos
            pastphi = bitdic[key[1:]]
            ii = int(key[0])
            temp = np.dot(m[0, ii], pastphi)
            bitdic[key] = temp
            phi[num * fac::step] = temp
            psi[num * fac::step] = np.dot(m[1, ii], pastphi)
        prevkeys = newkeys
    return (x, phi, psi)