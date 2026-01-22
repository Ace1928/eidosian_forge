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
def expectile(a, alpha=0.5, *, weights=None):
    """Compute the expectile at the specified level.

    Expectiles are a generalization of the expectation in the same way as
    quantiles are a generalization of the median. The expectile at level
    `alpha = 0.5` is the mean (average). See Notes for more details.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose expectile is desired.
    alpha : float, default: 0.5
        The level of the expectile; e.g., `alpha=0.5` gives the mean.
    weights : array_like, optional
        An array of weights associated with the values in `a`.
        The `weights` must be broadcastable to the same shape as `a`.
        Default is None, which gives each value a weight of 1.0.
        An integer valued weight element acts like repeating the corresponding
        observation in `a` that many times. See Notes for more details.

    Returns
    -------
    expectile : ndarray
        The empirical expectile at level `alpha`.

    See Also
    --------
    numpy.mean : Arithmetic average
    numpy.quantile : Quantile

    Notes
    -----
    In general, the expectile at level :math:`\\alpha` of a random variable
    :math:`X` with cumulative distribution function (CDF) :math:`F` is given
    by the unique solution :math:`t` of:

    .. math::

        \\alpha E((X - t)_+) = (1 - \\alpha) E((t - X)_+) \\,.

    Here, :math:`(x)_+ = \\max(0, x)` is the positive part of :math:`x`.
    This equation can be equivalently written as:

    .. math::

        \\alpha \\int_t^\\infty (x - t)\\mathrm{d}F(x)
        = (1 - \\alpha) \\int_{-\\infty}^t (t - x)\\mathrm{d}F(x) \\,.

    The empirical expectile at level :math:`\\alpha` (`alpha`) of a sample
    :math:`a_i` (the array `a`) is defined by plugging in the empirical CDF of
    `a`. Given sample or case weights :math:`w` (the array `weights`), it
    reads :math:`F_a(x) = \\frac{1}{\\sum_i w_i} \\sum_i w_i 1_{a_i \\leq x}`
    with indicator function :math:`1_{A}`. This leads to the definition of the
    empirical expectile at level `alpha` as the unique solution :math:`t` of:

    .. math::

        \\alpha \\sum_{i=1}^n w_i (a_i - t)_+ =
            (1 - \\alpha) \\sum_{i=1}^n w_i (t - a_i)_+ \\,.

    For :math:`\\alpha=0.5`, this simplifies to the weighted average.
    Furthermore, the larger :math:`\\alpha`, the larger the value of the
    expectile.

    As a final remark, the expectile at level :math:`\\alpha` can also be
    written as a minimization problem. One often used choice is

    .. math::

        \\operatorname{argmin}_t
        E(\\lvert 1_{t\\geq X} - \\alpha\\rvert(t - X)^2) \\,.

    References
    ----------
    .. [1] W. K. Newey and J. L. Powell (1987), "Asymmetric Least Squares
           Estimation and Testing," Econometrica, 55, 819-847.
    .. [2] T. Gneiting (2009). "Making and Evaluating Point Forecasts,"
           Journal of the American Statistical Association, 106, 746 - 762.
           :doi:`10.48550/arXiv.0912.0902`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import expectile
    >>> a = [1, 4, 2, -1]
    >>> expectile(a, alpha=0.5) == np.mean(a)
    True
    >>> expectile(a, alpha=0.2)
    0.42857142857142855
    >>> expectile(a, alpha=0.8)
    2.5714285714285716
    >>> weights = [1, 3, 1, 1]

    """
    if alpha < 0 or alpha > 1:
        raise ValueError('The expectile level alpha must be in the range [0, 1].')
    a = np.asarray(a)
    if weights is not None:
        weights = np.broadcast_to(weights, a.shape)

    def first_order(t):
        return np.average(np.abs((a <= t) - alpha) * (t - a), weights=weights)
    if alpha >= 0.5:
        x0 = np.average(a, weights=weights)
        x1 = np.amax(a)
    else:
        x1 = np.average(a, weights=weights)
        x0 = np.amin(a)
    if x0 == x1:
        return x0
    res = root_scalar(first_order, x0=x0, x1=x1)
    return res.root