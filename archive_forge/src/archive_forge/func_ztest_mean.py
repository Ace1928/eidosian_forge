import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def ztest_mean(self, value=0, alternative='two-sided'):
    """z-test of Null hypothesis that mean is equal to value.

        The alternative hypothesis H1 is defined by the following
        'two-sided': H1: mean not equal to value
        'larger' :   H1: mean larger than value
        'smaller' :  H1: mean smaller than value

        Parameters
        ----------
        value : float or array
            the hypothesized value for the mean
        alternative : str
            The alternative hypothesis, H1, has to be one of the following

              'two-sided': H1: mean not equal to value (default)
              'larger' :   H1: mean larger than value
              'smaller' :  H1: mean smaller than value

        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the t-test

        Notes
        -----
        This uses the same degrees of freedom correction as the t-test in the
        calculation of the standard error of the mean, i.e it uses
        `(sum_weights - 1)` instead of `sum_weights` in the denominator.
        See Examples below for the difference.

        Examples
        --------

        z-test on a proportion, with 20 observations, 15 of those are our event

        >>> import statsmodels.api as sm
        >>> x1 = [0, 1]
        >>> w1 = [5, 15]
        >>> d1 = sm.stats.DescrStatsW(x1, w1)
        >>> d1.ztest_mean(0.5)
        (2.5166114784235836, 0.011848940928347452)

        This differs from the proportions_ztest because of the degrees of
        freedom correction:
        >>> sm.stats.proportions_ztest(15, 20.0, value=0.5)
        (2.5819888974716112, 0.009823274507519247).

        We can replicate the results from ``proportions_ztest`` if we increase
        the weights to have artificially one more observation:

        >>> sm.stats.DescrStatsW(x1, np.array(w1)*21./20).ztest_mean(0.5)
        (2.5819888974716116, 0.0098232745075192366)
        """
    tstat = (self.mean - value) / self.std_mean
    if alternative == 'two-sided':
        pvalue = stats.norm.sf(np.abs(tstat)) * 2
    elif alternative == 'larger':
        pvalue = stats.norm.sf(tstat)
    elif alternative == 'smaller':
        pvalue = stats.norm.cdf(tstat)
    return (tstat, pvalue)