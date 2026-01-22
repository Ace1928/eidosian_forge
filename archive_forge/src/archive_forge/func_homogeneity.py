import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
def homogeneity(self, method='stuart_maxwell'):
    """
        Compare row and column marginal distributions.

        Parameters
        ----------
        method : str
            Either 'stuart_maxwell' or 'bhapkar', leading to two different
            estimates of the covariance matrix for the estimated
            difference between the row margins and the column margins.

        Returns
        -------
        Bunch
            A bunch with attributes:

            * statistic : float
                The chi^2 test statistic
            * pvalue : float
                The p-value of the test statistic
            * df : int
                The degrees of freedom of the reference distribution

        Notes
        -----
        For a 2x2 table this is equivalent to McNemar's test.  More
        generally the procedure tests the null hypothesis that the
        marginal distribution of the row factor is equal to the
        marginal distribution of the column factor.  For this to be
        meaningful, the two factors must have the same sample space
        (i.e. the same categories).
        """
    if self.table.shape[0] < 1:
        raise ValueError('table is empty')
    elif self.table.shape[0] == 1:
        b = _Bunch()
        b.statistic = 0
        b.pvalue = 1
        b.df = 0
        return b
    method = method.lower()
    if method not in ['bhapkar', 'stuart_maxwell']:
        raise ValueError("method '%s' for homogeneity not known" % method)
    n_obs = self.table.sum()
    pr = self.table.astype(np.float64) / n_obs
    row = pr.sum(1)[0:-1]
    col = pr.sum(0)[0:-1]
    pr = pr[0:-1, 0:-1]
    d = col - row
    df = pr.shape[0]
    if method == 'bhapkar':
        vmat = -(pr + pr.T) - np.outer(d, d)
        dv = col + row - 2 * np.diag(pr) - d ** 2
        np.fill_diagonal(vmat, dv)
    elif method == 'stuart_maxwell':
        vmat = -(pr + pr.T)
        dv = row + col - 2 * np.diag(pr)
        np.fill_diagonal(vmat, dv)
    try:
        statistic = n_obs * np.dot(d, np.linalg.solve(vmat, d))
    except np.linalg.LinAlgError:
        warnings.warn('Unable to invert covariance matrix', sm_exceptions.SingularMatrixWarning)
        b = _Bunch()
        b.statistic = np.nan
        b.pvalue = np.nan
        b.df = df
        return b
    pvalue = 1 - stats.chi2.cdf(statistic, df)
    b = _Bunch()
    b.statistic = statistic
    b.pvalue = pvalue
    b.df = df
    return b