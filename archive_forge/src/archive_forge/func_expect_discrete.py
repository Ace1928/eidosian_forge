from statsmodels.compat.python import lmap
import numpy as np
from scipy import stats, optimize, integrate
def expect_discrete(self, fn=None, args=(), loc=0, lb=None, ub=None, conditional=False):
    """calculate expected value of a function with respect to the distribution
    for discrete distribution

    Parameters
    ----------
        (self : distribution instance as defined in scipy stats)
        fn : function (default: identity mapping)
           Function for which integral is calculated. Takes only one argument.
        args : tuple
           argument (parameters) of the distribution
        optional keyword parameters
        lb, ub : numbers
           lower and upper bound for integration, default is set to the support
           of the distribution, lb and ub are inclusive (ul<=k<=ub)
        conditional : bool (False)
           If true then the expectation is corrected by the conditional
           probability of the integration interval. The return value is the
           expectation of the function, conditional on being in the given
           interval (k such that ul<=k<=ub).

    Returns
    -------
        expected value : float

    Notes
    -----
    * function is not vectorized
    * accuracy: uses self.moment_tol as stopping criterium
        for heavy tailed distribution e.g. zipf(4), accuracy for
        mean, variance in example is only 1e-5,
        increasing precision (moment_tol) makes zipf very slow
    * suppnmin=100 internal parameter for minimum number of points to evaluate
        could be added as keyword parameter, to evaluate functions with
        non-monotonic shapes, points include integers in (-suppnmin, suppnmin)
    * uses maxcount=1000 limits the number of points that are evaluated
        to break loop for infinite sums
        (a maximum of suppnmin+1000 positive plus suppnmin+1000 negative integers
        are evaluated)


    """
    maxcount = 1000
    suppnmin = 100
    if fn is None:

        def fun(x):
            return (x + loc) * self._pmf(x, *args)
    else:

        def fun(x):
            return fn(x + loc) * self._pmf(x, *args)
    self._argcheck(*args)
    if lb is None:
        lb = self.a
    else:
        lb = lb - loc
    if ub is None:
        ub = self.b
    else:
        ub = ub - loc
    if conditional:
        invfac = self.sf(lb, *args) - self.sf(ub + 1, *args)
    else:
        invfac = 1.0
    tot = 0.0
    low, upp = (self._ppf(0.001, *args), self._ppf(0.999, *args))
    low = max(min(-suppnmin, low), lb)
    upp = min(max(suppnmin, upp), ub)
    supp = np.arange(low, upp + 1, self.inc)
    tot = np.sum(fun(supp))
    diff = 1e+100
    pos = upp + self.inc
    count = 0
    while pos <= ub and diff > self.moment_tol and (count <= maxcount):
        diff = fun(pos)
        tot += diff
        pos += self.inc
        count += 1
    if self.a < 0:
        diff = 1e+100
        pos = low - self.inc
        while pos >= lb and diff > self.moment_tol and (count <= maxcount):
            diff = fun(pos)
            tot += diff
            pos -= self.inc
            count += 1
    if count > maxcount:
        print('sum did not converge')
    return tot / invfac