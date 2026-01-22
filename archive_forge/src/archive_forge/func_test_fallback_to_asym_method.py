import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import get_rdataset
from statsmodels.datasets.tests.test_utils import IGNORED_EXCEPTIONS
import statsmodels.stats.dist_dependence_measures as ddm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_fallback_to_asym_method(self):
    match_text = 'The asymptotic approximation will be used'
    with pytest.warns(UserWarning, match=match_text):
        statistic, pval, _ = ddm.distance_covariance_test(self.x, self.y, method='emp', B=200)
        assert_almost_equal(statistic, self.test_stat_emp_exp, 0)
        assert_almost_equal(pval, self.pval_asym_exp, 3)