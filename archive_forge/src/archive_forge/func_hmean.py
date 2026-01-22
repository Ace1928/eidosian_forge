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
def hmean(a, axis=0, dtype=None, *, weights=None):
    """Calculate the weighted harmonic mean along the specified axis.

    The weighted harmonic mean of the array :math:`a_i` associated to weights
    :math:`w_i` is:

    .. math::

        \\frac{ \\sum_{i=1}^n w_i }{ \\sum_{i=1}^n \\frac{w_i}{a_i} } \\, ,

    and, with equal weights, it gives:

    .. math::

        \\frac{ n }{ \\sum_{i=1}^n \\frac{1}{a_i} } \\, .

    Parameters
    ----------
    a : array_like
        Input array, masked array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the harmonic mean is computed. Default is 0.
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

        .. versionadded:: 1.9

    Returns
    -------
    hmean : ndarray
        See `dtype` parameter above.

    See Also
    --------
    numpy.mean : Arithmetic average
    numpy.average : Weighted average
    gmean : Geometric mean

    Notes
    -----
    The harmonic mean is computed over a single dimension of the input
    array, axis=0 by default, or all values in the array if axis=None.
    float64 intermediate and return values are used for integer inputs.

    References
    ----------
    .. [1] "Weighted Harmonic Mean", *Wikipedia*,
           https://en.wikipedia.org/wiki/Harmonic_mean#Weighted_harmonic_mean
    .. [2] Ferger, F., "The nature and use of the harmonic mean", Journal of
           the American Statistical Association, vol. 26, pp. 36-40, 1931

    Examples
    --------
    >>> from scipy.stats import hmean
    >>> hmean([1, 4])
    1.6000000000000001
    >>> hmean([1, 2, 3, 4, 5, 6, 7])
    2.6997245179063363
    >>> hmean([1, 4, 7], weights=[3, 1, 3])
    1.9029126213592233

    """
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
            return 1.0 / np.average(1.0 / a, axis=axis, weights=weights)
    else:
        raise ValueError('Harmonic mean only defined if all elements greater than or equal to zero')