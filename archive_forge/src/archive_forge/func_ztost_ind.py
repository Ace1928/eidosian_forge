import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def ztost_ind(self, low, upp, usevar='pooled'):
    """
        test of equivalence for two independent samples, based on z-test

        Parameters
        ----------
        low, upp : float
            equivalence interval low < m1 - m2 < upp
        usevar : str, 'pooled' or 'unequal'
            If ``pooled``, then the standard deviation of the samples is assumed to be
            the same. If ``unequal``, then Welch ttest with Satterthwait degrees
            of freedom is used

        Returns
        -------
        pvalue : float
            pvalue of the non-equivalence test
        t1, pv1 : tuple of floats
            test statistic and pvalue for lower threshold test
        t2, pv2 : tuple of floats
            test statistic and pvalue for upper threshold test
        """
    tt1 = self.ztest_ind(alternative='larger', usevar=usevar, value=low)
    tt2 = self.ztest_ind(alternative='smaller', usevar=usevar, value=upp)
    return (np.maximum(tt1[1], tt2[1]), tt1, tt2)