from typing import Collection, Dict, List, Union, overload, Iterable
from typing_extensions import Literal
import msgpack
from pkg_resources import resource_filename
import numpy as np
from numpy.typing import ArrayLike
from .._cache import cache
from .._typing import _FloatLike_co
@cache(level=10)
def interval_frequencies(n_bins: int, *, fmin: _FloatLike_co, intervals: Union[str, Collection[float]], bins_per_octave: int=12, tuning: float=0.0, sort: bool=True) -> np.ndarray:
    """Construct a set of frequencies from an interval set

    Parameters
    ----------
    n_bins : int
        The number of frequencies to generate

    fmin : float > 0
        The minimum frequency

    intervals : str or array of floats in [1, 2)
        If `str`, must be one of the following:
        - `'equal'` - equal temperament
        - `'pythagorean'` - Pythagorean intervals
        - `'ji3'` - 3-limit just intonation
        - `'ji5'` - 5-limit just intonation
        - `'ji7'` - 7-limit just intonation

        Otherwise, an array of intervals in the range [1, 2) can be provided.

    bins_per_octave : int > 0
        If `intervals` is a string specification, how many bins to
        generate per octave.
        If `intervals` is an array, then this parameter is ignored.

    tuning : float
        Deviation from A440 tuning in fractional bins.
        This is only used when `intervals == 'equal'`

    sort : bool
        Sort the intervals in ascending order.

    Returns
    -------
    frequencies : array of float
        The frequencies

    Examples
    --------
    Generate two octaves of Pythagorean intervals starting at 55Hz

    >>> librosa.interval_frequencies(24, fmin=55, intervals="pythagorean", bins_per_octave=12)
    array([ 55.   ,  58.733,  61.875,  66.075,  69.609,  74.334,  78.311,
            82.5  ,  88.099,  92.812,  99.112, 104.414, 110.   , 117.466,
           123.75 , 132.149, 139.219, 148.668, 156.621, 165.   , 176.199,
           185.625, 198.224, 208.828])

    Generate two octaves of 5-limit intervals starting at 55Hz

    >>> librosa.interval_frequencies(24, fmin=55, intervals="ji5", bins_per_octave=12)
    array([ 55.   ,  58.667,  61.875,  66.   ,  68.75 ,  73.333,  77.344,
            82.5  ,  88.   ,  91.667,  99.   , 103.125, 110.   , 117.333,
           123.75 , 132.   , 137.5  , 146.667, 154.687, 165.   , 176.   ,
           183.333, 198.   , 206.25 ])

    Generate three octaves using only three intervals

    >>> intervals = [1, 4/3, 3/2]
    >>> librosa.interval_frequencies(9, fmin=55, intervals=intervals)
    array([ 55.   ,  73.333,  82.5  , 110.   , 146.667, 165.   , 220.   ,
       293.333, 330.   ])
    """
    if isinstance(intervals, str):
        if intervals == 'equal':
            ratios = 2.0 ** ((tuning + np.arange(0, bins_per_octave, dtype=float)) / bins_per_octave)
        elif intervals == 'pythagorean':
            ratios = pythagorean_intervals(bins_per_octave=bins_per_octave, sort=sort)
        elif intervals == 'ji3':
            ratios = plimit_intervals(primes=[3], bins_per_octave=bins_per_octave, sort=sort)
        elif intervals == 'ji5':
            ratios = plimit_intervals(primes=[3, 5], bins_per_octave=bins_per_octave, sort=sort)
        elif intervals == 'ji7':
            ratios = plimit_intervals(primes=[3, 5, 7], bins_per_octave=bins_per_octave, sort=sort)
    else:
        ratios = np.array(intervals)
        bins_per_octave = len(ratios)
    n_octaves = np.ceil(n_bins / bins_per_octave)
    all_ratios = np.multiply.outer(2.0 ** np.arange(n_octaves), ratios).flatten()[:n_bins]
    if sort:
        all_ratios = np.sort(all_ratios)
    return all_ratios * fmin