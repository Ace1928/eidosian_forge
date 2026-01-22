import numpy as np
from scipy import linalg
from scipy.linalg import toeplitz
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.datasets import sunspots
from statsmodels.tsa.arima_process import (
def getpoly(self, params):
    ar = np.r_[[1], -params[:self.nar]]
    ma = np.r_[[1], params[-self.nma:]]
    import numpy.polynomial as poly
    return (poly.Polynomial(ar), poly.Polynomial(ma))