import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
@cache_readonly
def bse_fe(self):
    """
        Returns the standard errors of the fixed effect regression
        coefficients.
        """
    p = self.model.exog.shape[1]
    return np.sqrt(np.diag(self.cov_params())[0:p])