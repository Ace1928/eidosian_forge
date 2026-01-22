from numpy.testing import assert_almost_equal
import pytest
from statsmodels.datasets import stackloss
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from .results.el_results import RegressionResults
@pytest.mark.slow
def test_ci_beta3(self):
    beta3ci = self.res1.conf_int_el(3, method='nm')
    assert_almost_equal(beta3ci, self.res2.test_ci_beta3, 6)