import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
@cache_readonly
def sumsquares(self):
    """weighted sum of squares of demeaned data"""
    return np.dot((self.demeaned ** 2).T, self.weights)