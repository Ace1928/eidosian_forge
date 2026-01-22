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
@_deprecate_positional_args(version='1.14')
def comb(N, k, *, exact=False, repetition=False, legacy=_NoValue):
    """The number of combinations of N things taken k at a time.

    This is often expressed as "N choose k".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        For integers, if `exact` is False, then floating point precision is
        used, otherwise the result is computed exactly. For non-integers, if
        `exact` is True, is disregarded.
    repetition : bool, optional
        If `repetition` is True, then the number of combinations with
        repetition is computed.
    legacy : bool, optional
        If `legacy` is True and `exact` is True, then non-integral arguments
        are cast to ints; if `legacy` is False, the result for non-integral
        arguments is unaffected by the value of `exact`.

        .. deprecated:: 1.9.0
            Using `legacy` is deprecated and will removed by
            Scipy 1.14.0. If you want to keep the legacy behaviour, cast
            your inputs directly, e.g.
            ``comb(int(your_N), int(your_k), exact=True)``.

    Returns
    -------
    val : int, float, ndarray
        The total number of combinations.

    See Also
    --------
    binom : Binomial coefficient considered as a function of two real
            variables.

    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If N < 0, or k < 0, then 0 is returned.
    - If k > N and repetition=False, then 0 is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import comb
    >>> k = np.array([3, 4])
    >>> n = np.array([10, 10])
    >>> comb(n, k, exact=False)
    array([ 120.,  210.])
    >>> comb(10, 3, exact=True)
    120
    >>> comb(10, 3, exact=True, repetition=True)
    220

    """
    if legacy is not _NoValue:
        warnings.warn("Using 'legacy' keyword is deprecated and will be removed by Scipy 1.14.0. If you want to keep the legacy behaviour, cast your inputs directly, e.g. 'comb(int(your_N), int(your_k), exact=True)'.", DeprecationWarning, stacklevel=2)
    if repetition:
        return comb(N + k - 1, k, exact=exact, legacy=legacy)
    if exact:
        if int(N) == N and int(k) == k:
            return _comb_int(N, k)
        elif legacy:
            return _comb_int(N, k)
        return comb(N, k)
    else:
        k, N = (asarray(k), asarray(N))
        cond = (k <= N) & (N >= 0) & (k >= 0)
        vals = binom(N, k)
        if isinstance(vals, np.ndarray):
            vals[~cond] = 0
        elif not cond:
            vals = np.float64(0)
        return vals