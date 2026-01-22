import numpy as np
import numpy.linalg as L
from scipy.linalg import solveh_banded
from scipy.optimize import golden
from models import _hbspline     #removed because this was segfaulting
import warnings
def fit_target_df(self, y, x=None, df=None, weights=None, tol=0.001, apen=0, bpen=0.001):
    """
        Fit smoothing spline with approximately df degrees of freedom
        used in the fit, i.e. so that self.trace() is approximately df.

        Uses binary search strategy.

        In general, df must be greater than the dimension of the null space
        of the Gram inner product. For cubic smoothing splines, this means
        that df > 2.

        INPUTS:
           y       -- response variable
           x       -- if None, uses self.x
           df      -- target degrees of freedom
           weights -- optional array of weights
           tol     -- (relative) tolerance for convergence
           apen    -- lower bound of penalty for binary search
           bpen    -- upper bound of penalty for binary search

        OUTPUTS: None
           The smoothing spline is determined by self.coef,
           subsequent calls of __call__ will be the smoothing spline.
        """
    df = df or self.target_df
    olddf = y.shape[0] - self.m
    if hasattr(self, 'pen'):
        self.fit(y, x=x, weights=weights, pen=self.pen)
        curdf = self.trace()
        if np.fabs(curdf - df) / df < tol:
            return
        if curdf > df:
            apen, bpen = (self.pen, 2 * self.pen)
        else:
            apen, bpen = (0.0, self.pen)
    while True:
        curpen = 0.5 * (apen + bpen)
        self.fit(y, x=x, weights=weights, pen=curpen)
        curdf = self.trace()
        if curdf > df:
            apen, bpen = (curpen, 2 * curpen)
        else:
            apen, bpen = (apen, curpen)
        if apen >= self.penmax:
            raise ValueError('penalty too large, try setting penmax                    higher or decreasing df')
        if np.fabs(curdf - df) / df < tol:
            break