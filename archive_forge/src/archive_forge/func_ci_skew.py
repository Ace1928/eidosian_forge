import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def ci_skew(self, sig=0.05, upper_bound=None, lower_bound=None):
    """
        Returns the confidence interval for skewness.

        Parameters
        ----------
        sig : float
            The significance level.  Default is .05

        upper_bound : float
            Maximum value of skewness the upper limit can be.
            Default is .99 confidence limit assuming normality.

        lower_bound : float
            Minimum value of skewness the lower limit can be.
            Default is .99 confidence level assuming normality.

        Returns
        -------
        Interval : tuple
            Confidence interval for the skewness

        Notes
        -----
        If function returns f(a) and f(b) must have different signs, consider
        expanding lower and upper bounds
        """
    nobs = self.nobs
    endog = self.endog
    if upper_bound is None:
        upper_bound = skew(endog) + 2.5 * (6.0 * nobs * (nobs - 1.0) / ((nobs - 2.0) * (nobs + 1.0) * (nobs + 3.0))) ** 0.5
    if lower_bound is None:
        lower_bound = skew(endog) - 2.5 * (6.0 * nobs * (nobs - 1.0) / ((nobs - 2.0) * (nobs + 1.0) * (nobs + 3.0))) ** 0.5
    self.r0 = chi2.ppf(1 - sig, 1)
    llim = optimize.brentq(self._ci_limits_skew, lower_bound, skew(endog))
    ulim = optimize.brentq(self._ci_limits_skew, skew(endog), upper_bound)
    return (llim, ulim)