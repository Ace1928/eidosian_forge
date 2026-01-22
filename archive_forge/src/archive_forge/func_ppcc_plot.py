from __future__ import annotations
import math
import warnings
from collections import namedtuple
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py
from ._fit import FitResult
from ._stats_py import find_repeats, _normtest_finish, SignificanceResult
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr
from ._axis_nan_policy import _axis_nan_policy_factory
def ppcc_plot(x, a, b, dist='tukeylambda', plot=None, N=80):
    """Calculate and optionally plot probability plot correlation coefficient.

    The probability plot correlation coefficient (PPCC) plot can be used to
    determine the optimal shape parameter for a one-parameter family of
    distributions.  It cannot be used for distributions without shape
    parameters
    (like the normal distribution) or with multiple shape parameters.

    By default a Tukey-Lambda distribution (`stats.tukeylambda`) is used. A
    Tukey-Lambda PPCC plot interpolates from long-tailed to short-tailed
    distributions via an approximately normal one, and is therefore
    particularly useful in practice.

    Parameters
    ----------
    x : array_like
        Input array.
    a, b : scalar
        Lower and upper bounds of the shape parameter to use.
    dist : str or stats.distributions instance, optional
        Distribution or distribution function name.  Objects that look enough
        like a stats.distributions instance (i.e. they have a ``ppf`` method)
        are also accepted.  The default is ``'tukeylambda'``.
    plot : object, optional
        If given, plots PPCC against the shape parameter.
        `plot` is an object that has to have methods "plot" and "text".
        The `matplotlib.pyplot` module or a Matplotlib Axes object can be used,
        or a custom object with the same methods.
        Default is None, which means that no plot is created.
    N : int, optional
        Number of points on the horizontal axis (equally distributed from
        `a` to `b`).

    Returns
    -------
    svals : ndarray
        The shape values for which `ppcc` was calculated.
    ppcc : ndarray
        The calculated probability plot correlation coefficient values.

    See Also
    --------
    ppcc_max, probplot, boxcox_normplot, tukeylambda

    References
    ----------
    J.J. Filliben, "The Probability Plot Correlation Coefficient Test for
    Normality", Technometrics, Vol. 17, pp. 111-117, 1975.

    Examples
    --------
    First we generate some random data from a Weibull distribution
    with shape parameter 2.5, and plot the histogram of the data:

    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> c = 2.5
    >>> x = stats.weibull_min.rvs(c, scale=4, size=2000, random_state=rng)

    Take a look at the histogram of the data.

    >>> fig1, ax = plt.subplots(figsize=(9, 4))
    >>> ax.hist(x, bins=50)
    >>> ax.set_title('Histogram of x')
    >>> plt.show()

    Now we explore this data with a PPCC plot as well as the related
    probability plot and Box-Cox normplot.  A red line is drawn where we
    expect the PPCC value to be maximal (at the shape parameter ``c``
    used above):

    >>> fig2 = plt.figure(figsize=(12, 4))
    >>> ax1 = fig2.add_subplot(1, 3, 1)
    >>> ax2 = fig2.add_subplot(1, 3, 2)
    >>> ax3 = fig2.add_subplot(1, 3, 3)
    >>> res = stats.probplot(x, plot=ax1)
    >>> res = stats.boxcox_normplot(x, -4, 4, plot=ax2)
    >>> res = stats.ppcc_plot(x, c/2, 2*c, dist='weibull_min', plot=ax3)
    >>> ax3.axvline(c, color='r')
    >>> plt.show()

    """
    if b <= a:
        raise ValueError('`b` has to be larger than `a`.')
    svals = np.linspace(a, b, num=N)
    ppcc = np.empty_like(svals)
    for k, sval in enumerate(svals):
        _, r2 = probplot(x, sval, dist=dist, fit=True)
        ppcc[k] = r2[-1]
    if plot is not None:
        plot.plot(svals, ppcc, 'x')
        _add_axis_labels_title(plot, xlabel='Shape Values', ylabel='Prob Plot Corr. Coef.', title='(%s) PPCC Plot' % dist)
    return (svals, ppcc)