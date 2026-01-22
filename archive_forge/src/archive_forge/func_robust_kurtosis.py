from scipy import stats
import numpy as np
from statsmodels.tools.sm_exceptions import ValueWarning
def robust_kurtosis(y, axis=0, ab=(5.0, 50.0), dg=(2.5, 25.0), excess=True):
    """
    Calculates the four kurtosis measures in Kim & White

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
    axis : int or None, optional
        Axis along which the kurtosis are computed.  If `None`, the
        entire array is used.
    a iterable, optional
        Contains 100*(alpha, beta) in the kr3 measure where alpha is the tail
        quantile cut-off for measuring the extreme tail and beta is the central
        quantile cutoff for the standardization of the measure
    db : iterable, optional
        Contains 100*(delta, gamma) in the kr4 measure where delta is the tail
        quantile for measuring extreme values and gamma is the central quantile
        used in the the standardization of the measure
    excess : bool, optional
        If true (default), computed values are excess of those for a standard
        normal distribution.

    Returns
    -------
    kr1 : ndarray
          The standard kurtosis estimator.
    kr2 : ndarray
          Kurtosis estimator based on octiles.
    kr3 : ndarray
          Kurtosis estimators based on exceedance expectations.
    kr4 : ndarray
          Kurtosis measure based on the spread between high and low quantiles.

    Notes
    -----
    The robust kurtosis measures are defined

    .. math::

        KR_{2}=\\frac{\\left(\\hat{q}_{.875}-\\hat{q}_{.625}\\right)
        +\\left(\\hat{q}_{.375}-\\hat{q}_{.125}\\right)}
        {\\hat{q}_{.75}-\\hat{q}_{.25}}

    .. math::

        KR_{3}=\\frac{\\hat{E}\\left(y|y>\\hat{q}_{1-\\alpha}\\right)
        -\\hat{E}\\left(y|y<\\hat{q}_{\\alpha}\\right)}
        {\\hat{E}\\left(y|y>\\hat{q}_{1-\\beta}\\right)
        -\\hat{E}\\left(y|y<\\hat{q}_{\\beta}\\right)}

    .. math::

        KR_{4}=\\frac{\\hat{q}_{1-\\delta}-\\hat{q}_{\\delta}}
        {\\hat{q}_{1-\\gamma}-\\hat{q}_{\\gamma}}

    where :math:`\\hat{q}_{p}` is the estimated quantile at :math:`p`.

    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of
       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
       March 2004.
    """
    if axis is None or (y.squeeze().ndim == 1 and y.ndim != 1):
        y = y.ravel()
        axis = 0
    alpha, beta = ab
    delta, gamma = dg
    perc = (12.5, 25.0, 37.5, 62.5, 75.0, 87.5, delta, 100.0 - delta, gamma, 100.0 - gamma)
    e1, e2, e3, e5, e6, e7, fd, f1md, fg, f1mg = np.percentile(y, perc, axis=axis)
    expected_value = expected_robust_kurtosis(ab, dg) if excess else np.zeros(4)
    kr1 = stats.kurtosis(y, axis, False) - expected_value[0]
    kr2 = (e7 - e5 + (e3 - e1)) / (e6 - e2) - expected_value[1]
    if y.ndim == 1:
        kr3 = _kr3(y, alpha, beta)
    else:
        kr3 = np.apply_along_axis(_kr3, axis, y, alpha, beta)
    kr3 -= expected_value[2]
    kr4 = (f1md - fd) / (f1mg - fg) - expected_value[3]
    return (kr1, kr2, kr3, kr4)