import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def ci_var(self, lower_bound=None, upper_bound=None, sig=0.05):
    """
        Returns the confidence interval for the variance.

        Parameters
        ----------
        lower_bound : float
            The minimum value the lower confidence interval can
            take. The p-value from test_var(lower_bound) must be lower
            than 1 - significance level. Default is .99 confidence
            limit assuming normality

        upper_bound : float
            The maximum value the upper confidence interval
            can take. The p-value from test_var(upper_bound) must be lower
            than 1 - significance level.  Default is .99 confidence
            limit assuming normality

        sig : float
            The significance level. Default is .05

        Returns
        -------
        Interval : tuple
            Confidence interval for the variance

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> random_numbers = np.random.standard_normal(100)
        >>> el_analysis = sm.emplike.DescStat(random_numbers)
        >>> el_analysis.ci_var()
        (0.7539322567470305, 1.229998852496268)
        >>> el_analysis.ci_var(.5, 2)
        (0.7539322567469926, 1.2299988524962664)

        Notes
        -----
        If the function returns the error f(a) and f(b) must have
        different signs, consider lowering lower_bound and raising
        upper_bound.
        """
    endog = self.endog
    if upper_bound is None:
        upper_bound = (self.nobs - 1) * endog.var() / chi2.ppf(0.0001, self.nobs - 1)
    if lower_bound is None:
        lower_bound = (self.nobs - 1) * endog.var() / chi2.ppf(0.9999, self.nobs - 1)
    self.r0 = chi2.ppf(1 - sig, 1)
    llim = optimize.brentq(self._ci_limits_var, lower_bound, endog.var())
    ulim = optimize.brentq(self._ci_limits_var, endog.var(), upper_bound)
    return (llim, ulim)