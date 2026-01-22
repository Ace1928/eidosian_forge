import numpy as np
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import (
def confint_lintransf(self, const=-1, slope=2, alpha=0.05, alternative='two-sided'):
    """confidence interval of a linear transformation of prob1

        This computes the confidence interval for

            d = const + slope * prob1

        Default values correspond to Somers' d.

        Parameters
        ----------
        const, slope : float
            Constant and slope for linear (affine) transformation.
        alpha : float
            Significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : str
            The alternative hypothesis, H1, has to be one of the following

               * 'two-sided' : H1: ``prob - value`` not equal to 0.
               * 'larger' :   H1: ``prob - value > 0``
               * 'smaller' :  H1: ``prob - value < 0``

        Returns
        -------
        lower : float or ndarray
            Lower confidence limit. This is -inf for the one-sided alternative
            "smaller".
        upper : float or ndarray
            Upper confidence limit. This is inf for the one-sided alternative
            "larger".

        """
    low_p, upp_p = self.conf_int(alpha=alpha, alternative=alternative)
    low = const + slope * low_p
    upp = const + slope * upp_p
    if slope < 0:
        low, upp = (upp, low)
    return (low, upp)