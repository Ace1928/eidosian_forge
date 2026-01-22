import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat
from .results.el_results import DescStatRes
def test_test_mean(self):
    assert_almost_equal(self.res1.test_mean(14), self.res2.test_mean_14, 4)