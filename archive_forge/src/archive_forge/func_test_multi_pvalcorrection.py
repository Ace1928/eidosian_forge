import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
def test_multi_pvalcorrection(self):
    res_multtest = self.res2
    pval0 = res_multtest[:, 0]
    pvalscorr = np.sort(fdrcorrection(pval0, method='n')[1])
    assert_almost_equal(pvalscorr, res_multtest[:, 7], 15)
    pvalscorr = np.sort(fdrcorrection(pval0, method='i')[1])
    assert_almost_equal(pvalscorr, res_multtest[:, 6], 15)