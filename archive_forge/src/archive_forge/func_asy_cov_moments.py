import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
@cache_readonly
def asy_cov_moments(self):
    """

        `sqrt(T) * g_T(b_0) asy N(K delta, V)`

        mean is not implemented,
        V is the same as cov_moments in __init__ argument
        """
    return self.cov_moments