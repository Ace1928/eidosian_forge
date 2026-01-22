import numpy as np
import pandas as pd
from scipy import stats
def t_test(self, value=0, alternative='two-sided'):
    """
        z- or t-test for hypothesis that mean is equal to value

        Parameters
        ----------
        value : array_like
            value under the null hypothesis
        alternative : str
            'two-sided', 'larger', 'smaller'

        Returns
        -------
        stat : ndarray
            test statistic
        pvalue : ndarray
            p-value of the hypothesis test, the distribution is given by
            the attribute of the instance, specified in `__init__`. Default
            if not specified is the normal distribution.
        """
    stat = (self.predicted_mean - value) / self.se_mean
    if alternative in ['two-sided', '2-sided', '2s']:
        pvalue = self.dist.sf(np.abs(stat), *self.dist_args) * 2
    elif alternative in ['larger', 'l']:
        pvalue = self.dist.sf(stat, *self.dist_args)
    elif alternative in ['smaller', 's']:
        pvalue = self.dist.cdf(stat, *self.dist_args)
    else:
        raise ValueError('invalid alternative')
    return (stat, pvalue)