from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, cast
import numpy as np
import numpy.typing as npt
import math
import warnings
from collections import namedtuple
from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn
from scipy._lib.deprecation import (_NoValue, _deprecate_positional_args,
def newton_cotes(rn, equal=0):
    """
    Return weights and error coefficient for Newton-Cotes integration.

    Suppose we have (N+1) samples of f at the positions
    x_0, x_1, ..., x_N. Then an N-point Newton-Cotes formula for the
    integral between x_0 and x_N is:

    :math:`\\int_{x_0}^{x_N} f(x)dx = \\Delta x \\sum_{i=0}^{N} a_i f(x_i)
    + B_N (\\Delta x)^{N+2} f^{N+1} (\\xi)`

    where :math:`\\xi \\in [x_0,x_N]`
    and :math:`\\Delta x = \\frac{x_N-x_0}{N}` is the average samples spacing.

    If the samples are equally-spaced and N is even, then the error
    term is :math:`B_N (\\Delta x)^{N+3} f^{N+2}(\\xi)`.

    Parameters
    ----------
    rn : int
        The integer order for equally-spaced data or the relative positions of
        the samples with the first sample at 0 and the last at N, where N+1 is
        the length of `rn`. N is the order of the Newton-Cotes integration.
    equal : int, optional
        Set to 1 to enforce equally spaced data.

    Returns
    -------
    an : ndarray
        1-D array of weights to apply to the function at the provided sample
        positions.
    B : float
        Error coefficient.

    Notes
    -----
    Normally, the Newton-Cotes rules are used on smaller integration
    regions and a composite rule is used to return the total integral.

    Examples
    --------
    Compute the integral of sin(x) in [0, :math:`\\pi`]:

    >>> from scipy.integrate import newton_cotes
    >>> import numpy as np
    >>> def f(x):
    ...     return np.sin(x)
    >>> a = 0
    >>> b = np.pi
    >>> exact = 2
    >>> for N in [2, 4, 6, 8, 10]:
    ...     x = np.linspace(a, b, N + 1)
    ...     an, B = newton_cotes(N, 1)
    ...     dx = (b - a) / N
    ...     quad = dx * np.sum(an * f(x))
    ...     error = abs(quad - exact)
    ...     print('{:2d}  {:10.9f}  {:.5e}'.format(N, quad, error))
    ...
     2   2.094395102   9.43951e-02
     4   1.998570732   1.42927e-03
     6   2.000017814   1.78136e-05
     8   1.999999835   1.64725e-07
    10   2.000000001   1.14677e-09

    """
    try:
        N = len(rn) - 1
        if equal:
            rn = np.arange(N + 1)
        elif np.all(np.diff(rn) == 1):
            equal = 1
    except Exception:
        N = rn
        rn = np.arange(N + 1)
        equal = 1
    if equal and N in _builtincoeffs:
        na, da, vi, nb, db = _builtincoeffs[N]
        an = na * np.array(vi, dtype=float) / da
        return (an, float(nb) / db)
    if rn[0] != 0 or rn[-1] != N:
        raise ValueError('The sample positions must start at 0 and end at N')
    yi = rn / float(N)
    ti = 2 * yi - 1
    nvec = np.arange(N + 1)
    C = ti ** nvec[:, np.newaxis]
    Cinv = np.linalg.inv(C)
    for i in range(2):
        Cinv = 2 * Cinv - Cinv.dot(C).dot(Cinv)
    vec = 2.0 / (nvec[::2] + 1)
    ai = Cinv[:, ::2].dot(vec) * (N / 2.0)
    if N % 2 == 0 and equal:
        BN = N / (N + 3.0)
        power = N + 2
    else:
        BN = N / (N + 2.0)
        power = N + 1
    BN = BN - np.dot(yi ** power, ai)
    p1 = power + 1
    fac = power * math.log(N) - gammaln(p1)
    fac = math.exp(fac)
    return (ai, BN * fac)