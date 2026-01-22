import numpy as np
from scipy import stats
from scipy.special import comb
import warnings
from statsmodels.tools.validation import array_like
def runs_prob_odd(self, r):
    n0, n1 = (self.n0, self.n1)
    k = (r + 1) // 2
    tmp0 = comb(n0 - 1, k - 1)
    tmp1 = comb(n1 - 1, k - 2)
    tmp3 = comb(n0 - 1, k - 2)
    tmp4 = comb(n1 - 1, k - 1)
    return (tmp0 * tmp1 + tmp3 * tmp4) / self.comball