import numpy as np
from scipy import stats
from scipy.special import comb
from scipy.stats.distributions import rv_continuous
import matplotlib.pyplot as plt
from numpy import where, inf
from numpy import abs as np_abs
def meanexcess_dist(self, lb, *args, **kwds):
    if np.ndim(lb) == 0:
        return self.expect(lb=lb, conditional=True)
    else:
        return np.array([self.expect(lb=lbb, conditional=True) for lbb in lb])