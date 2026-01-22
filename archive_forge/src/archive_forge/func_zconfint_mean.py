import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def zconfint_mean(self, alpha=0.05, alternative='two-sided'):
    """two-sided confidence interval for weighted mean of data

        Confidence interval is based on normal distribution.
        If the data is 2d, then these are separate confidence intervals
        for each column.

        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following

              'two-sided': H1: mean not equal to value (default)
              'larger' :   H1: mean larger than value
              'smaller' :  H1: mean smaller than value

        Returns
        -------
        lower, upper : floats or ndarrays
            lower and upper bound of confidence interval

        Notes
        -----
        In a previous version, statsmodels 0.4, alpha was the confidence
        level, e.g. 0.95
        """
    return _zconfint_generic(self.mean, self.std_mean, alpha, alternative)