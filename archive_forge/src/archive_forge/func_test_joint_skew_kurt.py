import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat
from .results.el_results import DescStatRes
def test_joint_skew_kurt(self):
    assert_almost_equal(self.res1.test_joint_skew_kurt(0, 0), self.res2.test_joint_skew_kurt, 4)