import warnings
import numpy as np
import scipy.stats._stats_py
from . import distributions
from .._lib._bunch import _make_tuple_bunch
from ._stats_pythran import siegelslopes as siegelslopes_pythran
def linregress(x, y=None, alternative='two-sided'):
    """
    Calculate a linear least-squares regression for two sets of measurements.

    Parameters
    ----------
    x, y : array_like
        Two sets of measurements.  Both arrays should have the same length.  If
        only `x` is given (and ``y=None``), then it must be a two-dimensional
        array where one dimension has length 2.  The two sets of measurements
        are then found by splitting the array along the length-2 dimension. In
        the case where ``y=None`` and `x` is a 2x2 array, ``linregress(x)`` is
        equivalent to ``linregress(x[0], x[1])``.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the slope of the regression line is nonzero
        * 'less': the slope of the regression line is less than zero
        * 'greater':  the slope of the regression line is greater than zero

        .. versionadded:: 1.7.0

    Returns
    -------
    result : ``LinregressResult`` instance
        The return value is an object with the following attributes:

        slope : float
            Slope of the regression line.
        intercept : float
            Intercept of the regression line.
        rvalue : float
            The Pearson correlation coefficient. The square of ``rvalue``
            is equal to the coefficient of determination.
        pvalue : float
            The p-value for a hypothesis test whose null hypothesis is
            that the slope is zero, using Wald Test with t-distribution of
            the test statistic. See `alternative` above for alternative
            hypotheses.
        stderr : float
            Standard error of the estimated slope (gradient), under the
            assumption of residual normality.
        intercept_stderr : float
            Standard error of the estimated intercept, under the assumption
            of residual normality.

    See Also
    --------
    scipy.optimize.curve_fit :
        Use non-linear least squares to fit a function to data.
    scipy.optimize.leastsq :
        Minimize the sum of squares of a set of equations.

    Notes
    -----
    Missing values are considered pair-wise: if a value is missing in `x`,
    the corresponding value in `y` is masked.

    For compatibility with older versions of SciPy, the return value acts
    like a ``namedtuple`` of length 5, with fields ``slope``, ``intercept``,
    ``rvalue``, ``pvalue`` and ``stderr``, so one can continue to write::

        slope, intercept, r, p, se = linregress(x, y)

    With that style, however, the standard error of the intercept is not
    available.  To have access to all the computed values, including the
    standard error of the intercept, use the return value as an object
    with attributes, e.g.::

        result = linregress(x, y)
        print(result.intercept, result.intercept_stderr)

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> rng = np.random.default_rng()

    Generate some data:

    >>> x = rng.random(10)
    >>> y = 1.6*x + rng.random(10)

    Perform the linear regression:

    >>> res = stats.linregress(x, y)

    Coefficient of determination (R-squared):

    >>> print(f"R-squared: {res.rvalue**2:.6f}")
    R-squared: 0.717533

    Plot the data along with the fitted line:

    >>> plt.plot(x, y, 'o', label='original data')
    >>> plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
    >>> plt.legend()
    >>> plt.show()

    Calculate 95% confidence interval on slope and intercept:

    >>> # Two-sided inverse Students t-distribution
    >>> # p - probability, df - degrees of freedom
    >>> from scipy.stats import t
    >>> tinv = lambda p, df: abs(t.ppf(p/2, df))

    >>> ts = tinv(0.05, len(x)-2)
    >>> print(f"slope (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}")
    slope (95%): 1.453392 +/- 0.743465
    >>> print(f"intercept (95%): {res.intercept:.6f}"
    ...       f" +/- {ts*res.intercept_stderr:.6f}")
    intercept (95%): 0.616950 +/- 0.544475

    """
    TINY = 1e-20
    if y is None:
        x = np.asarray(x)
        if x.shape[0] == 2:
            x, y = x
        elif x.shape[1] == 2:
            x, y = x.T
        else:
            raise ValueError(f'If only `x` is given as input, it has to be of shape (2, N) or (N, 2); provided shape was {x.shape}.')
    else:
        x = np.asarray(x)
        y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        raise ValueError('Inputs must not be empty.')
    if np.amax(x) == np.amin(x) and len(x) > 1:
        raise ValueError('Cannot calculate a linear regression if all x values are identical')
    n = len(x)
    xmean = np.mean(x, None)
    ymean = np.mean(y, None)
    ssxm, ssxym, _, ssym = np.cov(x, y, bias=1).flat
    if ssxm == 0.0 or ssym == 0.0:
        r = 0.0
    else:
        r = ssxym / np.sqrt(ssxm * ssym)
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0
    slope = ssxym / ssxm
    intercept = ymean - slope * xmean
    if n == 2:
        if y[0] == y[1]:
            prob = 1.0
        else:
            prob = 0.0
        slope_stderr = 0.0
        intercept_stderr = 0.0
    else:
        df = n - 2
        t = r * np.sqrt(df / ((1.0 - r + TINY) * (1.0 + r + TINY)))
        t, prob = scipy.stats._stats_py._ttest_finish(df, t, alternative)
        slope_stderr = np.sqrt((1 - r ** 2) * ssym / ssxm / df)
        intercept_stderr = slope_stderr * np.sqrt(ssxm + xmean ** 2)
    return LinregressResult(slope=slope, intercept=intercept, rvalue=r, pvalue=prob, stderr=slope_stderr, intercept_stderr=intercept_stderr)