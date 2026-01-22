from typing import Collection, Dict, List, Union, overload, Iterable
from typing_extensions import Literal
import msgpack
from pkg_resources import resource_filename
import numpy as np
from numpy.typing import ArrayLike
from .._cache import cache
from .._typing import _FloatLike_co
@cache(level=10)
def pythagorean_intervals(*, bins_per_octave: int=12, sort: bool=True, return_factors: bool=False) -> Union[np.ndarray, List[Dict[int, int]]]:
    """Pythagorean intervals

    Intervals are constructed by stacking ratios of 3/2 (i.e.,
    just perfect fifths) and folding down to a single octave::

        1, 3/2, 9/8, 27/16, 81/64, ...

    Note that this differs from 3-limit just intonation intervals
    in that Pythagorean intervals only use positive powers of 3
    (ascending fifths) while 3-limit intervals use both positive
    and negative powers (descending fifths).

    Parameters
    ----------
    bins_per_octave : int
        The number of intervals to generate
    sort : bool
        If `True` then intervals are returned in ascending order.
        If `False`, then intervals are returned in circle-of-fifths order.
    return_factors : bool
        If `True` then return a list of dictionaries encoding the prime factorization
        of each interval as `{2: p2, 3: p3}` (meaning `3**p3 * 2**p2`).
        If `False` (default), return intervals as an array of floating point numbers.

    Returns
    -------
    intervals : np.ndarray or list of dictionaries
        The constructed interval set. All intervals are mapped
        to the range [1, 2).

    See Also
    --------
    plimit_intervals

    Examples
    --------
    Generate the first 12 intervals

    >>> librosa.pythagorean_intervals(bins_per_octave=12)
    array([1.      , 1.067871, 1.125   , 1.201355, 1.265625, 1.351524,
           1.423828, 1.5     , 1.601807, 1.6875  , 1.802032, 1.898437])
    >>> # Compare to the 12-tone equal temperament intervals:
    >>> 2**(np.arange(12)/12)
    array([1.      , 1.059463, 1.122462, 1.189207, 1.259921, 1.33484 ,
           1.414214, 1.498307, 1.587401, 1.681793, 1.781797, 1.887749])

    Or the first 7, in circle-of-fifths order

    >>> librosa.pythagorean_intervals(bins_per_octave=7, sort=False)
    array([1.      , 1.5     , 1.125   , 1.6875  , 1.265625, 1.898437,
           1.423828])

    Generate the first 7, in circle-of-fifths other and factored form

    >>> librosa.pythagorean_intervals(bins_per_octave=7, sort=False, return_factors=True)
    [
        {2: 0, 3: 0},
        {2: -1, 3: 1},
        {2: -3, 3: 2},
        {2: -4, 3: 3},
        {2: -6, 3: 4},
        {2: -7, 3: 5},
        {2: -9, 3: 6}
    ]
    """
    pow3 = np.arange(bins_per_octave)
    log_ratios: np.ndarray
    pow2: np.ndarray
    log_ratios, pow2 = np.modf(pow3 * np.log2(3))
    too_small = log_ratios < 0
    log_ratios[too_small] += 1
    pow2[too_small] += 1
    pow2 = pow2.astype(int)
    idx: Iterable[int]
    if sort:
        idx = np.argsort(log_ratios)
        log_ratios = log_ratios[idx]
    else:
        idx = range(bins_per_octave)
    if return_factors:
        return list(({2: -pow2[i], 3: pow3[i]} for i in idx))
    return np.power(2, log_ratios)