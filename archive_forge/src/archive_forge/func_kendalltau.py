import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def kendalltau(x, y, use_ties=True, use_missing=False, method='auto', alternative='two-sided'):
    """
    Computes Kendall's rank correlation tau on two variables *x* and *y*.

    Parameters
    ----------
    x : sequence
        First data list (for example, time).
    y : sequence
        Second data list.
    use_ties : {True, False}, optional
        Whether ties correction should be performed.
    use_missing : {False, True}, optional
        Whether missing data should be allocated a rank of 0 (False) or the
        average rank (True)
    method : {'auto', 'asymptotic', 'exact'}, optional
        Defines which method is used to calculate the p-value [1]_.
        'asymptotic' uses a normal approximation valid for large samples.
        'exact' computes the exact p-value, but can only be used if no ties
        are present. As the sample size increases, the 'exact' computation
        time may grow and the result may lose some precision.
        'auto' is the default and selects the appropriate
        method based on a trade-off between speed and accuracy.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the rank correlation is nonzero
        * 'less': the rank correlation is negative (less than zero)
        * 'greater':  the rank correlation is positive (greater than zero)

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : float
           The tau statistic.
        pvalue : float
           The p-value for a hypothesis test whose null hypothesis is
           an absence of association, tau = 0.

    References
    ----------
    .. [1] Maurice G. Kendall, "Rank Correlation Methods" (4th Edition),
           Charles Griffin & Co., 1970.

    """
    x, y, n = _chk_size(x, y)
    x, y = (x.flatten(), y.flatten())
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    if m is not nomask:
        x = ma.array(x, mask=m, copy=True)
        y = ma.array(y, mask=m, copy=True)
        n -= int(m.sum())
    if n < 2:
        res = scipy.stats._stats_py.SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res
    rx = ma.masked_equal(rankdata(x, use_missing=use_missing), 0)
    ry = ma.masked_equal(rankdata(y, use_missing=use_missing), 0)
    idx = rx.argsort()
    rx, ry = (rx[idx], ry[idx])
    C = np.sum([((ry[i + 1:] > ry[i]) * (rx[i + 1:] > rx[i])).filled(0).sum() for i in range(len(ry) - 1)], dtype=float)
    D = np.sum([((ry[i + 1:] < ry[i]) * (rx[i + 1:] > rx[i])).filled(0).sum() for i in range(len(ry) - 1)], dtype=float)
    xties = count_tied_groups(x)
    yties = count_tied_groups(y)
    if use_ties:
        corr_x = np.sum([v * k * (k - 1) for k, v in xties.items()], dtype=float)
        corr_y = np.sum([v * k * (k - 1) for k, v in yties.items()], dtype=float)
        denom = ma.sqrt((n * (n - 1) - corr_x) / 2.0 * (n * (n - 1) - corr_y) / 2.0)
    else:
        denom = n * (n - 1) / 2.0
    tau = (C - D) / denom
    if method == 'exact' and (xties or yties):
        raise ValueError('Ties found, exact method cannot be used.')
    if method == 'auto':
        if (not xties and (not yties)) and (n <= 33 or min(C, n * (n - 1) / 2.0 - C) <= 1):
            method = 'exact'
        else:
            method = 'asymptotic'
    if not xties and (not yties) and (method == 'exact'):
        prob = _kendall_p_exact(n, C, alternative)
    elif method == 'asymptotic':
        var_s = n * (n - 1) * (2 * n + 5)
        if use_ties:
            var_s -= np.sum([v * k * (k - 1) * (2 * k + 5) * 1.0 for k, v in xties.items()])
            var_s -= np.sum([v * k * (k - 1) * (2 * k + 5) * 1.0 for k, v in yties.items()])
            v1 = np.sum([v * k * (k - 1) for k, v in xties.items()], dtype=float) * np.sum([v * k * (k - 1) for k, v in yties.items()], dtype=float)
            v1 /= 2.0 * n * (n - 1)
            if n > 2:
                v2 = np.sum([v * k * (k - 1) * (k - 2) for k, v in xties.items()], dtype=float) * np.sum([v * k * (k - 1) * (k - 2) for k, v in yties.items()], dtype=float)
                v2 /= 9.0 * n * (n - 1) * (n - 2)
            else:
                v2 = 0
        else:
            v1 = v2 = 0
        var_s /= 18.0
        var_s += v1 + v2
        z = (C - D) / np.sqrt(var_s)
        _, prob = scipy.stats._stats_py._normtest_finish(z, alternative)
    else:
        raise ValueError('Unknown method ' + str(method) + ' specified, please use auto, exact or asymptotic.')
    res = scipy.stats._stats_py.SignificanceResult(tau, prob)
    res.correlation = tau
    return res