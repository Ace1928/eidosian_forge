import numpy as np
from numpy.testing import assert_allclose
from statsmodels.datasets.cpunish import load
from statsmodels.discrete.discrete_model import (
import statsmodels.discrete.tests.results.results_count_margins as res_stata
from statsmodels.tools.tools import add_constant
def test_margins_table(self):
    res1 = self.res1
    sl = self.res1_slice
    rf = self.rtol_fac
    assert_allclose(self.margeff.margeff, self.res1.params[sl], rtol=1e-05 * rf)
    assert_allclose(self.margeff.margeff_se, self.res1.bse[sl], rtol=1e-06 * rf)
    assert_allclose(self.margeff.pvalues, self.res1.pvalues[sl], rtol=5e-06 * rf)
    assert_allclose(self.margeff.conf_int(), res1.margins_table[sl, 4:6], rtol=1e-06 * rf)