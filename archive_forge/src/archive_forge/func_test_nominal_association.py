import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
def test_nominal_association(self):
    """
        Assess independence for nominal factors.

        Assessment of independence between rows and columns using
        chi^2 testing.  The rows and columns are treated as nominal
        (unordered) categorical variables.

        Returns
        -------
        A bunch containing the following attributes:

        statistic : float
            The chi^2 test statistic.
        df : int
            The degrees of freedom of the reference distribution
        pvalue : float
            The p-value for the test.
        """
    statistic = np.asarray(self.chi2_contribs).sum()
    df = np.prod(np.asarray(self.table.shape) - 1)
    pvalue = 1 - stats.chi2.cdf(statistic, df)
    b = _Bunch()
    b.statistic = statistic
    b.df = df
    b.pvalue = pvalue
    return b