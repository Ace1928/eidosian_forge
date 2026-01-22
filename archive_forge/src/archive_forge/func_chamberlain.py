import numpy as np
import warnings
import scipy.stats as stats
from numpy.linalg import pinv
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import (RegressionModel,
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
def chamberlain(n, q, alpha=0.05):
    return norm.ppf(1 - alpha / 2) * np.sqrt(q * (1 - q) / n)