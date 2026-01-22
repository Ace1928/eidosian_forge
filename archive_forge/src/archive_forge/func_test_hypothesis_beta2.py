from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import stackloss
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from .results.el_results import RegressionResults
@pytest.mark.slow
def test_hypothesis_beta2(self):
    beta2res = self.res1.el_test([1], [2], return_weights=1, method='nm')
    assert_almost_equal(beta2res[:2], self.res2.test_beta2[:2], 4)
    assert_almost_equal(beta2res[2], self.res2.test_beta2[2], 4)