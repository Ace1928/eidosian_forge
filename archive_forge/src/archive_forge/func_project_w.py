import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
@cache_readonly
def project_w(self):
    moments_deriv = self.moments_deriv
    res = moments_deriv.dot(self.asy_transf_params)
    res += np.eye(res.shape[0])
    return res