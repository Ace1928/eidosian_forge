import numpy as np
from scipy import stats
from statsmodels.compat.scipy import multivariate_t
from statsmodels.distributions.copula.copulas import Copula
def dependence_tail(self, corr=None):
    """
        Bivariate tail dependence parameter.

        Joe (2014) p. 182

        Parameters
        ----------
        corr : None or float
            Pearson correlation. If corr is None, then the correlation will be
            taken from the copula attribute.

        Returns
        -------
        Lower and upper tail dependence coefficients of the copula with given
        Pearson correlation coefficient.
        """
    if corr is None:
        corr = self.corr
    if corr.shape == (2, 2):
        corr = corr[0, 1]
    df = self.df
    t = -np.sqrt((df + 1) * (1 - corr) / 1 + corr)
    lam = 2 * stats.t.cdf(t, df + 1)
    return (lam, lam)