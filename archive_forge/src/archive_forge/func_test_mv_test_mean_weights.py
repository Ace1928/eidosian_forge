import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat
from .results.el_results import DescStatRes
def test_mv_test_mean_weights(self):
    assert_almost_equal(self.mvres1.mv_test_mean(np.array([14, 56]), return_weights=1)[2], self.res2.mv_test_mean_wts, 4)