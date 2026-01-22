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
@_axis_nan_policy_factory(lambda x: x, n_samples=1, n_outputs=1, too_small=0, paired=True, result_to_tuple=lambda x: (x,), kwd_samples=['weights'])
def pmean(a, p, *, axis=0, dtype=None, weights=None):
    """Calculate the weighted power mean along the specified axis.

    The weighted power mean of the array :math:`a_i` associated to weights
    :math:`w_i` is:

    .. math::

        \\left( \\frac{ \\sum_{i=1}^n w_i a_i^p }{ \\sum_{i=1}^n w_i }
              \\right)^{ 1 / p } \\, ,

    and, with equal weights, it gives:

    .. math::

        \\left( \\frac{ 1 }{ n } \\sum_{i=1}^n a_i^p \\right)^{ 1 / p }  \\, .

    When ``p=0``, it returns the geometric mean.

    This mean is also called generalized mean or Hölder mean, and must not be
    confused with the Kolmogorov generalized mean, also called
    quasi-arithmetic mean or generalized f-mean [3]_.

    Parameters
    ----------
    a : array_like
        Input array, masked array or object that can be converted to an array.
    p : int or float
        Exponent.
    axis : int or None, optional
        Axis along which the power mean is computed. Default is 0.
        If None, compute over the whole array `a`.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed. If `dtype` is not specified, it defaults to the
        dtype of `a`, unless `a` has an integer `dtype` with a precision less
        than that of the default platform integer. In that case, the default
        platform integer is used.
    weights : array_like, optional
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given `axis`) or of the same shape as `a`.
        Default is None, which gives each value a weight of 1.0.

    Returns
    -------
    pmean : ndarray, see `dtype` parameter above.
        Output array containing the power mean values.

    See Also
    --------
    numpy.average : Weighted average
    gmean : Geometric mean
    hmean : Harmonic mean

    Notes
    -----
    The power mean is computed over a single dimension of the input
    array, ``axis=0`` by default, or all values in the array if ``axis=None``.
    float64 intermediate and return values are used for integer inputs.

    .. versionadded:: 1.9

    References
    ----------
    .. [1] "Generalized Mean", *Wikipedia*,
           https://en.wikipedia.org/wiki/Generalized_mean
    .. [2] Norris, N., "Convexity properties of generalized mean value
           functions", The Annals of Mathematical Statistics, vol. 8,
           pp. 118-120, 1937
    .. [3] Bullen, P.S., Handbook of Means and Their Inequalities, 2003

    Examples
    --------
    >>> from scipy.stats import pmean, hmean, gmean
    >>> pmean([1, 4], 1.3)
    2.639372938300652
    >>> pmean([1, 2, 3, 4, 5, 6, 7], 1.3)
    4.157111214492084
    >>> pmean([1, 4, 7], -2, weights=[3, 1, 3])
    1.4969684896631954

    For p=-1, power mean is equal to harmonic mean:

    >>> pmean([1, 4, 7], -1, weights=[3, 1, 3])
    1.9029126213592233
    >>> hmean([1, 4, 7], weights=[3, 1, 3])
    1.9029126213592233

    For p=0, power mean is defined as the geometric mean:

    >>> pmean([1, 4, 7], 0, weights=[3, 1, 3])
    2.80668351922014
    >>> gmean([1, 4, 7], weights=[3, 1, 3])
    2.80668351922014

    """
    if not isinstance(p, (int, float)):
        raise ValueError('Power mean only defined for exponent of type int or float.')
    if p == 0:
        return gmean(a, axis=axis, dtype=dtype, weights=weights)
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=dtype)
    elif dtype:
        if isinstance(a, np.ma.MaskedArray):
            a = np.ma.asarray(a, dtype=dtype)
        else:
            a = np.asarray(a, dtype=dtype)
    if np.all(a >= 0):
        if weights is not None:
            weights = np.asanyarray(weights, dtype=dtype)
        with np.errstate(divide='ignore'):
            return np.float_power(np.average(np.float_power(a, p), axis=axis, weights=weights), 1 / p)
    else:
        raise ValueError('Power mean only defined if all elements greater than or equal to zero')