import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import cancer
from statsmodels.emplike.originregress import ELOriginRegress
from .results.el_results import OriginResults
def test_llf(self):
    assert_almost_equal(self.res1.llf_el, self.res2.test_llf_hat, 4)