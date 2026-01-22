import math
import numpy as np
from scipy import linalg, stats, special
from .linalg_decomp_1 import SvdArray
def logpdf_obs(self, x):
    x = x - self.mean
    x_whitened = self.whiten(x)
    logdetsigma = np.log(np.linalg.det(sigma))
    sigma2 = 1.0
    llike = 0.5 * (np.log(sigma2) - 2.0 * np.log(np.diagonal(self.cholsigmainv)) + x_whitened ** 2 / sigma2 + np.log(2 * np.pi))
    return llike