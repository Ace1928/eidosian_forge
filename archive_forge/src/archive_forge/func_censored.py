import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
def censored(self, censor_val):
    self.d = (self.endog != censor_val) * 1.0
    ix = np.argsort(np.squeeze(self.endog))
    self.sortix = ix
    self.sortix_rev = np.zeros(ix.shape, int)
    self.sortix_rev[ix] = np.arange(len(ix))
    self.endog = np.squeeze(self.endog[ix])
    self.endog = _adjust_shape(self.endog, 1)
    self.exog = np.squeeze(self.exog[ix])
    self.d = np.squeeze(self.d[ix])
    self.W_in = np.empty((self.nobs, 1))
    for i in range(1, self.nobs + 1):
        P = 1
        for j in range(1, i):
            P *= ((self.nobs - j) / (float(self.nobs) - j + 1)) ** self.d[j - 1]
        self.W_in[i - 1, 0] = P * self.d[i - 1] / (float(self.nobs) - i + 1)