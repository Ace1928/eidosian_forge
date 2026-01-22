import numpy as np
from scipy._lib._util import _lazywhere
from scipy.special import gammaln
def logW(y, p, phi):
    alpha = _alpha(p)
    jmax = y ** (2 - p) / ((2 - p) * phi)
    logWmax = np.array((1 - alpha) * jmax)
    tol = logWmax - 37
    j = np.ceil(jmax)
    while (_logWj(y, np.ceil(j), p, phi) > tol).any():
        j = np.where(_logWj(y, j, p, phi) > tol, j + 1, j)
    j_u = j
    j = np.floor(jmax)
    j = np.where(j > 1, j, 1)
    while (_logWj(y, j, p, phi) > tol).any() and (j > 1).any():
        j = np.where(_logWj(y, j, p, phi) > tol, j - 1, 1)
    j_l = j
    sumw = _sumw(y, j_l, j_u, logWmax, p, phi)
    return logWmax + np.log(sumw)