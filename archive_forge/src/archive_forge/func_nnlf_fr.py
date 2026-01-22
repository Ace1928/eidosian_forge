from statsmodels.compat.python import lmap
import numpy as np
from scipy import stats, optimize, integrate
def nnlf_fr(self, thetash, x, frmask):
    try:
        if frmask is not None:
            theta = frmask.copy()
            theta[np.isnan(frmask)] = thetash
        else:
            theta = thetash
        loc = theta[-2]
        scale = theta[-1]
        args = tuple(theta[:-2])
    except IndexError:
        raise ValueError('Not enough input arguments.')
    if not self._argcheck(*args) or scale <= 0:
        return np.inf
    x = np.array((x - loc) / scale)
    cond0 = (x <= self.a) | (x >= self.b)
    if np.any(cond0):
        return np.inf
    else:
        N = len(x)
        return self._nnlf(x, *args) + N * np.log(scale)