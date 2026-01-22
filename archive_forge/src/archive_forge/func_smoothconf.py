from statsmodels.compat.python import lzip, lfilter
import numpy as np
import scipy.integrate
from scipy.special import factorial
from numpy import exp, multiply, square, divide, subtract, inf
def smoothconf(self, xs, ys, x, alpha=0.05):
    """Returns the kernel smoothing estimate with confidence 1sigma bounds
        """
    xs, ys = self.in_domain(xs, ys, x)
    if len(xs) > 0:
        fittedvals = np.array([self.smooth(xs, ys, xx) for xx in xs])
        sqresid = square(subtract(ys, fittedvals))
        w = np.sum(self((xs - x) / self.h))
        v = np.sum([rr * self((xx - x) / self.h) for xx, rr in zip(xs, sqresid)])
        var = v / w
        sd = np.sqrt(var)
        K = self.L2Norm
        yhat = self.smooth(xs, ys, x)
        from scipy import stats
        crit = stats.norm.isf(alpha / 2)
        err = crit * sd * np.sqrt(K) / np.sqrt(w * self.h * self.norm_const)
        return (yhat - err, yhat, yhat + err)
    else:
        return (np.nan, np.nan, np.nan)