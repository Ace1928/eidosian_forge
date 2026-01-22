import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import heart
from statsmodels.emplike.aft_el import emplikeAFT
from statsmodels.tools import add_constant
from .results.el_results import AFTRes
@pytest.mark.slow
def test_betaci(self):
    ci = self.res1.ci_beta(1, -0.06, 0)
    ll = ci[0]
    ul = ci[1]
    ll_pval = self.res1.test_beta([ll], [1])[1]
    ul_pval = self.res1.test_beta([ul], [1])[1]
    assert_almost_equal(ul_pval, 0.05, decimal=4)
    assert_almost_equal(ll_pval, 0.05, decimal=4)