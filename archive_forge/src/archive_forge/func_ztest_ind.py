import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def ztest_ind(self, alternative='two-sided', usevar='pooled', value=0):
    """z-test for the null hypothesis of identical means

        Parameters
        ----------
        x1 : array_like, 1-D or 2-D
            first of the two independent samples, see notes for 2-D case
        x2 : array_like, 1-D or 2-D
            second of the two independent samples, see notes for 2-D case
        alternative : str
            The alternative hypothesis, H1, has to be one of the following
            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value

        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then the standard deviations of the samples may
            be different.
        value : float
            difference between the means under the Null hypothesis.

        Returns
        -------
        tstat : float
            test statistic
        pvalue : float
            pvalue of the z-test

        """
    d1 = self.d1
    d2 = self.d2
    if usevar == 'pooled':
        stdm = self.std_meandiff_pooledvar
    elif usevar == 'unequal':
        stdm = self.std_meandiff_separatevar
    else:
        raise ValueError('usevar can only be "pooled" or "unequal"')
    tstat, pval = _zstat_generic(d1.mean, d2.mean, stdm, alternative, diff=value)
    return (tstat, pval)