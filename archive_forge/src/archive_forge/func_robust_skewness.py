from scipy import stats
import numpy as np
from statsmodels.tools.sm_exceptions import ValueWarning
def robust_skewness(y, axis=0):
    """
    Calculates the four skewness measures in Kim & White

    Parameters
    ----------
    y : array_like
        Data to compute use in the estimator.
    axis : int or None, optional
        Axis along which the skewness measures are computed.  If `None`, the
        entire array is used.

    Returns
    -------
    sk1 : ndarray
          The standard skewness estimator.
    sk2 : ndarray
          Skewness estimator based on quartiles.
    sk3 : ndarray
          Skewness estimator based on mean-median difference, standardized by
          absolute deviation.
    sk4 : ndarray
          Skewness estimator based on mean-median difference, standardized by
          standard deviation.

    Notes
    -----
    The robust skewness measures are defined

    .. math::

        SK_{2}=\\frac{\\left(q_{.75}-q_{.5}\\right)
        -\\left(q_{.5}-q_{.25}\\right)}{q_{.75}-q_{.25}}

    .. math::

        SK_{3}=\\frac{\\mu-\\hat{q}_{0.5}}
        {\\hat{E}\\left[\\left|y-\\hat{\\mu}\\right|\\right]}

    .. math::

        SK_{4}=\\frac{\\mu-\\hat{q}_{0.5}}{\\hat{\\sigma}}

    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of
       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,
       March 2004.
    """
    if axis is None:
        y = y.ravel()
        axis = 0
    y = np.sort(y, axis)
    q1, q2, q3 = np.percentile(y, [25.0, 50.0, 75.0], axis=axis)
    mu = y.mean(axis)
    shape = (y.size,)
    if axis is not None:
        shape = list(mu.shape)
        shape.insert(axis, 1)
        shape = tuple(shape)
    mu_b = np.reshape(mu, shape)
    q2_b = np.reshape(q2, shape)
    sigma = np.sqrt(np.mean((y - mu_b) ** 2, axis))
    sk1 = stats.skew(y, axis=axis)
    sk2 = (q1 + q3 - 2.0 * q2) / (q3 - q1)
    sk3 = (mu - q2) / np.mean(abs(y - q2_b), axis=axis)
    sk4 = (mu - q2) / sigma
    return (sk1, sk2, sk3, sk4)