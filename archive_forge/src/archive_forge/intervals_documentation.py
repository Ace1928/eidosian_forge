from typing import Collection, Dict, List, Union, overload, Iterable
from typing_extensions import Literal
import msgpack
from pkg_resources import resource_filename
import numpy as np
from numpy.typing import ArrayLike
from .._cache import cache
from .._typing import _FloatLike_co
Construct p-limit intervals for a given set of prime factors.

    This function is based on the "harmonic crystal growth" algorithm
    of [#1]_ [#2]_.

    .. [#1] Tenney, James.
        "On ‘Crystal Growth’ in harmonic space (1993–1998)."
        Contemporary Music Review 27.1 (2008): 47-56.

    .. [#2] Sabat, Marc, and James Tenney.
        "Three crystal growth algorithms in 23-limit constrained harmonic space."
        Contemporary Music Review 27, no. 1 (2008): 57-78.

    Parameters
    ----------
    primes : array of odd primes
        Which prime factors are to be used
    bins_per_octave : int
        The number of intervals to construct
    sort : bool
        If `True` then intervals are returned in ascending order.
        If `False`, then intervals are returned in crystal growth order.
    return_factors : bool
        If `True` then return a list of dictionaries encoding the prime factorization
        of each interval as `{2: p2, 3: p3, ...}` (meaning `3**p3 * 2**p2`).
        If `False` (default), return intervals as an array of floating point numbers.

    Returns
    -------
    intervals : np.ndarray or list of dictionaries
        The constructed interval set. All intervals are mapped
        to the range [1, 2).

    See Also
    --------
    pythagorean_intervals

    Examples
    --------
    Compare 3-limit tuning to Pythagorean tuning and 12-TET

    >>> librosa.plimit_intervals(primes=[3], bins_per_octave=12)
    array([1.        , 1.05349794, 1.125     , 1.18518519, 1.265625  ,
           1.33333333, 1.40466392, 1.5       , 1.58024691, 1.6875    ,
           1.77777778, 1.8984375 ])
    >>> # Pythagorean intervals:
    >>> librosa.pythagorean_intervals(bins_per_octave=12)
    array([1.        , 1.06787109, 1.125     , 1.20135498, 1.265625  ,
           1.35152435, 1.42382812, 1.5       , 1.60180664, 1.6875    ,
           1.80203247, 1.8984375 ])
    >>> # 12-TET intervals:
    >>> 2**(np.arange(12)/12)
    array([1.        , 1.05946309, 1.12246205, 1.18920712, 1.25992105,
           1.33483985, 1.41421356, 1.49830708, 1.58740105, 1.68179283,
           1.78179744, 1.88774863])

    Create a 7-bin, 5-limit interval set

    >>> librosa.plimit_intervals(primes=[3, 5], bins_per_octave=7)
    array([1.        , 1.125     , 1.25      , 1.33333333, 1.5       ,
           1.66666667, 1.875     ])

    The same example, but now in factored form

    >>> librosa.plimit_intervals(primes=[3, 5], bins_per_octave=7,
    ...                          return_factors=True)
    [
        {},
        {2: -3, 3: 2},
        {2: -2, 5: 1},
        {2: 2, 3: -1},
        {2: -1, 3: 1},
        {3: -1, 5: 1},
        {2: -3, 3: 1, 5: 1}
    ]
    