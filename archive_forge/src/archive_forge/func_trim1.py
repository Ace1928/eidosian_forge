import warnings
import math
from math import gcd
from collections import namedtuple
import numpy as np
from numpy import array, asarray, ma
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements
from scipy._lib._util import (check_random_state, MapWrapper, _get_nan,
import scipy.special as special
from scipy import linalg
from . import distributions
from . import _mstats_basic as mstats_basic
from ._stats_mstats_common import (_find_repeats, linregress, theilslopes,
from ._stats import (_kendall_dis, _toint64, _weightedrankedtau,
from dataclasses import dataclass, field
from ._hypotests import _all_partitions
from ._stats_pythran import _compute_outer_prob_inside_method
from ._resampling import (MonteCarloMethod, PermutationMethod, BootstrapMethod,
from ._axis_nan_policy import (_axis_nan_policy_factory,
from ._binomtest import _binary_search_for_binom_tst as _binary_search
from scipy._lib._bunch import _make_tuple_bunch
from scipy import stats
from scipy.optimize import root_scalar
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy._lib._util import normalize_axis_index
from scipy._lib._util import float_factorial  # noqa: F401
from scipy.stats._mstats_basic import (  # noqa: F401
def trim1(a, proportiontocut, tail='right', axis=0):
    """Slice off a proportion from ONE end of the passed array distribution.

    If `proportiontocut` = 0.1, slices off 'leftmost' or 'rightmost'
    10% of scores. The lowest or highest values are trimmed (depending on
    the tail).
    Slice off less if proportion results in a non-integer slice index
    (i.e. conservatively slices off `proportiontocut` ).

    Parameters
    ----------
    a : array_like
        Input array.
    proportiontocut : float
        Fraction to cut off of 'left' or 'right' of distribution.
    tail : {'left', 'right'}, optional
        Defaults to 'right'.
    axis : int or None, optional
        Axis along which to trim data. Default is 0. If None, compute over
        the whole array `a`.

    Returns
    -------
    trim1 : ndarray
        Trimmed version of array `a`. The order of the trimmed content is
        undefined.

    Examples
    --------
    Create an array of 10 values and trim 20% of its lowest values:

    >>> import numpy as np
    >>> from scipy import stats
    >>> a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> stats.trim1(a, 0.2, 'left')
    array([2, 4, 3, 5, 6, 7, 8, 9])

    Note that the elements of the input array are trimmed by value, but the
    output array is not necessarily sorted.

    The proportion to trim is rounded down to the nearest integer. For
    instance, trimming 25% of the values from an array of 10 values will
    return an array of 8 values:

    >>> b = np.arange(10)
    >>> stats.trim1(b, 1/4).shape
    (8,)

    Multidimensional arrays can be trimmed along any axis or across the entire
    array:

    >>> c = [2, 4, 6, 8, 0, 1, 3, 5, 7, 9]
    >>> d = np.array([a, b, c])
    >>> stats.trim1(d, 0.8, axis=0).shape
    (1, 10)
    >>> stats.trim1(d, 0.8, axis=1).shape
    (3, 2)
    >>> stats.trim1(d, 0.8, axis=None).shape
    (6,)

    """
    a = np.asarray(a)
    if axis is None:
        a = a.ravel()
        axis = 0
    nobs = a.shape[axis]
    if proportiontocut >= 1:
        return []
    if tail.lower() == 'right':
        lowercut = 0
        uppercut = nobs - int(proportiontocut * nobs)
    elif tail.lower() == 'left':
        lowercut = int(proportiontocut * nobs)
        uppercut = nobs
    atmp = np.partition(a, (lowercut, uppercut - 1), axis)
    sl = [slice(None)] * atmp.ndim
    sl[axis] = slice(lowercut, uppercut)
    return atmp[tuple(sl)]