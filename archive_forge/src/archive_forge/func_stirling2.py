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
def stirling2(N, K, *, exact=False):
    """Generate Stirling number(s) of the second kind.

    Stirling numbers of the second kind count the number of ways to
    partition a set with N elements into K non-empty subsets.

    The values this function returns are calculated using a dynamic
    program which avoids redundant computation across the subproblems
    in the solution. For array-like input, this implementation also 
    avoids redundant computation across the different Stirling number
    calculations.

    The numbers are sometimes denoted

    .. math::

        {N \\brace{K}}

    see [1]_ for details. This is often expressed-verbally-as
    "N subset K".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    K : int, ndarray
        Number of non-empty subsets taken.
    exact : bool, optional
        Uses dynamic programming (DP) with floating point
        numbers for smaller arrays and uses a second order approximation due to
        Temme for larger entries  of `N` and `K` that allows trading speed for
        accuracy. See [2]_ for a description. Temme approximation is used for
        values `n>50`. The max error from the DP has max relative error
        `4.5*10^-16` for `n<=50` and the max error from the Temme approximation
        has max relative error `5*10^-5` for `51 <= n < 70` and
        `9*10^-6` for `70 <= n < 101`. Note that these max relative errors will
        decrease further as `n` increases.

    Returns
    -------
    val : int, float, ndarray
        The number of partitions.

    See Also
    --------
    comb : The number of combinations of N things taken k at a time.

    Notes
    -----
    - If N < 0, or K < 0, then 0 is returned.
    - If K > N, then 0 is returned.

    The output type will always be `int` or ndarray of `object`.
    The input must contain either numpy or python integers otherwise a
    TypeError is raised.

    References
    ----------
    .. [1] R. L. Graham, D. E. Knuth and O. Patashnik, "Concrete
        Mathematics: A Foundation for Computer Science," Addison-Wesley
        Publishing Company, Boston, 1989. Chapter 6, page 258.

    .. [2] Temme, Nico M. "Asymptotic estimates of Stirling numbers."
        Studies in Applied Mathematics 89.3 (1993): 233-243.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import stirling2
    >>> k = np.array([3, -1, 3])
    >>> n = np.array([10, 10, 9])
    >>> stirling2(n, k)
    array([9330, 0, 3025], dtype=object)

    """
    output_is_scalar = np.isscalar(N) and np.isscalar(K)
    N, K = (asarray(N), asarray(K))
    if not np.issubdtype(N.dtype, np.integer):
        raise TypeError('Argument `N` must contain only integers')
    if not np.issubdtype(K.dtype, np.integer):
        raise TypeError('Argument `K` must contain only integers')
    if not exact:
        return _stirling2_inexact(N.astype(float), K.astype(float))
    nk_pairs = list(set([(n.take(0), k.take(0)) for n, k in np.nditer([N, K], ['refs_ok'])]))
    heapify(nk_pairs)
    snsk_vals = defaultdict(int)
    for pair in [(0, 0), (1, 1), (2, 1), (2, 2)]:
        snsk_vals[pair] = 1
    n_old, n_row = (2, [0, 1, 1])
    while nk_pairs:
        n, k = heappop(nk_pairs)
        if n < 2 or k > n or k <= 0:
            continue
        elif k == n or k == 1:
            snsk_vals[n, k] = 1
            continue
        elif n != n_old:
            num_iters = n - n_old
            while num_iters > 0:
                n_row.append(1)
                for j in range(len(n_row) - 2, 1, -1):
                    n_row[j] = n_row[j] * j + n_row[j - 1]
                num_iters -= 1
            snsk_vals[n, k] = n_row[k]
        else:
            snsk_vals[n, k] = n_row[k]
        n_old, n_row = (n, n_row)
    out_types = [object, object, object] if exact else [float, float, float]
    it = np.nditer([N, K, None], ['buffered', 'refs_ok'], [['readonly'], ['readonly'], ['writeonly', 'allocate']], op_dtypes=out_types)
    with it:
        while not it.finished:
            it[2] = snsk_vals[int(it[0]), int(it[1])]
            it.iternext()
        output = it.operands[2]
        if output_is_scalar:
            output = output.take(0)
    return output