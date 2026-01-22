import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import get_rdataset
from statsmodels.datasets.tests.test_utils import IGNORED_EXCEPTIONS
import statsmodels.stats.dist_dependence_measures as ddm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_statistic_value_asym_method(self):
    statistic, pval, method = ddm.distance_covariance_test(self.x, self.y, method='asym')
    assert method == 'asym'
    assert_almost_equal(statistic, self.test_stat_asym_exp, 4)
    assert_almost_equal(pval, self.pval_asym_exp, 3)