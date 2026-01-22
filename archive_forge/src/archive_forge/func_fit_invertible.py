import numpy as np
from scipy import linalg
from scipy.linalg import toeplitz
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.datasets import sunspots
from statsmodels.tsa.arima_process import (
def fit_invertible(self, *args, **kwds):
    res = self.fit(*args, **kwds)
    ma = np.r_[[1], res.params[self.nar:self.nar + self.nma]]
    mainv, wasinvertible = invertibleroots(ma)
    if not wasinvertible:
        start_params = res.params.copy()
        start_params[self.nar:self.nar + self.nma] = mainv[1:]
        res = self.fit(start_params=start_params)
    return res