import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
@cache_readonly
def cov_mom_constraints(self):
    return self.cov_params_all[self.k_params:, self.k_params:]