import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def chi2_contribs(self):
    """
        Returns the contributions to the chi^2 statistic for independence.

        The returned table contains the contribution of each cell to the chi^2
        test statistic for the null hypothesis that the rows and columns
        are independent.
        """
    return self.resid_pearson ** 2