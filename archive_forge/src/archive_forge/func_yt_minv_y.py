import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
def yt_minv_y(self, y):
    """xSigmainvx
        does not use stored cholesky yet
        """
    return np.dot(x, linalg.cho_solve(linalg.cho_factor(self.m), x))