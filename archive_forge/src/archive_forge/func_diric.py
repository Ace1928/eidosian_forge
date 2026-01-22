import operator
import numpy as np
import math
import warnings
from collections import defaultdict
from heapq import heapify, heappop
from numpy import (pi, asarray, floor, isscalar, iscomplex, real,
from . import _ufuncs
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma,
from . import _specfun
from ._comb import _comb_int
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def diric(x, n):
    """Periodic sinc function, also called the Dirichlet function.

    The Dirichlet function is defined as::

        diric(x, n) = sin(x * n/2) / (n * sin(x / 2)),

    where `n` is a positive integer.

    Parameters
    ----------
    x : array_like
        Input data
    n : int
        Integer defining the periodicity.

    Returns
    -------
    diric : ndarray

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-8*np.pi, 8*np.pi, num=201)
    >>> plt.figure(figsize=(8, 8));
    >>> for idx, n in enumerate([2, 3, 4, 9]):
    ...     plt.subplot(2, 2, idx+1)
    ...     plt.plot(x, special.diric(x, n))
    ...     plt.title('diric, n={}'.format(n))
    >>> plt.show()

    The following example demonstrates that `diric` gives the magnitudes
    (modulo the sign and scaling) of the Fourier coefficients of a
    rectangular pulse.

    Suppress output of values that are effectively 0:

    >>> np.set_printoptions(suppress=True)

    Create a signal `x` of length `m` with `k` ones:

    >>> m = 8
    >>> k = 3
    >>> x = np.zeros(m)
    >>> x[:k] = 1

    Use the FFT to compute the Fourier transform of `x`, and
    inspect the magnitudes of the coefficients:

    >>> np.abs(np.fft.fft(x))
    array([ 3.        ,  2.41421356,  1.        ,  0.41421356,  1.        ,
            0.41421356,  1.        ,  2.41421356])

    Now find the same values (up to sign) using `diric`. We multiply
    by `k` to account for the different scaling conventions of
    `numpy.fft.fft` and `diric`:

    >>> theta = np.linspace(0, 2*np.pi, m, endpoint=False)
    >>> k * special.diric(theta, k)
    array([ 3.        ,  2.41421356,  1.        , -0.41421356, -1.        ,
           -0.41421356,  1.        ,  2.41421356])
    """
    x, n = (asarray(x), asarray(n))
    n = asarray(n + (x - x))
    x = asarray(x + (n - n))
    if issubdtype(x.dtype, inexact):
        ytype = x.dtype
    else:
        ytype = float
    y = zeros(x.shape, ytype)
    if np.finfo(ytype).eps < 1e-18:
        minval = 1e-11
    elif np.finfo(ytype).eps < 1e-15:
        minval = 1e-07
    else:
        minval = 0.001
    mask1 = (n <= 0) | (n != floor(n))
    place(y, mask1, nan)
    x = x / 2
    denom = sin(x)
    mask2 = 1 - mask1 & (abs(denom) < minval)
    xsub = extract(mask2, x)
    nsub = extract(mask2, n)
    zsub = xsub / pi
    place(y, mask2, pow(-1, np.round(zsub) * (nsub - 1)))
    mask = 1 - mask1 & 1 - mask2
    xsub = extract(mask, x)
    nsub = extract(mask, n)
    dsub = extract(mask, denom)
    place(y, mask, sin(nsub * xsub) / (nsub * dsub))
    return y