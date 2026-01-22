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
@_axis_nan_policy_factory(lambda x: x, n_outputs=1, default_axis=None, result_to_tuple=lambda x: (x,))
def circstd(samples, high=2 * pi, low=0, axis=None, nan_policy='propagate', *, normalize=False):
    """
    Compute the circular standard deviation for samples assumed to be in the
    range [low to high].

    Parameters
    ----------
    samples : array_like
        Input array.
    high : float or int, optional
        High boundary for the sample range. Default is ``2*pi``.
    low : float or int, optional
        Low boundary for the sample range. Default is 0.
    normalize : boolean, optional
        If True, the returned value is equal to ``sqrt(-2*log(R))`` and does
        not depend on the variable units. If False (default), the returned
        value is scaled by ``((high-low)/(2*pi))``.

    Returns
    -------
    circstd : float
        Circular standard deviation.

    See Also
    --------
    circmean : Circular mean.
    circvar : Circular variance.

    Notes
    -----
    This uses a definition of circular standard deviation from [1]_.
    Essentially, the calculation is as follows.

    .. code-block:: python

        import numpy as np
        C = np.cos(samples).mean()
        S = np.sin(samples).mean()
        R = np.sqrt(C**2 + S**2)
        l = 2*np.pi / (high-low)
        circstd = np.sqrt(-2*np.log(R)) / l

    In the limit of small angles, it returns a number close to the 'linear'
    standard deviation.

    References
    ----------
    .. [1] Mardia, K. V. (1972). 2. In *Statistics of Directional Data*
       (pp. 18-24). Academic Press. :doi:`10.1016/C2013-0-07425-7`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import circstd
    >>> import matplotlib.pyplot as plt
    >>> samples_1 = np.array([0.072, -0.158, 0.077, 0.108, 0.286,
    ...                       0.133, -0.473, -0.001, -0.348, 0.131])
    >>> samples_2 = np.array([0.111, -0.879, 0.078, 0.733, 0.421,
    ...                       0.104, -0.136, -0.867,  0.012,  0.105])
    >>> circstd_1 = circstd(samples_1)
    >>> circstd_2 = circstd(samples_2)

    Plot the samples.

    >>> fig, (left, right) = plt.subplots(ncols=2)
    >>> for image in (left, right):
    ...     image.plot(np.cos(np.linspace(0, 2*np.pi, 500)),
    ...                np.sin(np.linspace(0, 2*np.pi, 500)),
    ...                c='k')
    ...     image.axis('equal')
    ...     image.axis('off')
    >>> left.scatter(np.cos(samples_1), np.sin(samples_1), c='k', s=15)
    >>> left.set_title(f"circular std: {np.round(circstd_1, 2)!r}")
    >>> right.plot(np.cos(np.linspace(0, 2*np.pi, 500)),
    ...            np.sin(np.linspace(0, 2*np.pi, 500)),
    ...            c='k')
    >>> right.scatter(np.cos(samples_2), np.sin(samples_2), c='k', s=15)
    >>> right.set_title(f"circular std: {np.round(circstd_2, 2)!r}")
    >>> plt.show()

    """
    samples, sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_mean = sin_samp.mean(axis)
    cos_mean = cos_samp.mean(axis)
    with np.errstate(invalid='ignore'):
        R = np.minimum(1, hypot(sin_mean, cos_mean))
    res = sqrt(-2 * log(R))
    if not normalize:
        res *= (high - low) / (2.0 * pi)
    return res