import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat
from .results.el_results import DescStatRes
def test_ci_skew_weights(self):
    assert_almost_equal(self.res1.test_skew(0, return_weights=1)[2], self.res2.test_skew_wts, 4)