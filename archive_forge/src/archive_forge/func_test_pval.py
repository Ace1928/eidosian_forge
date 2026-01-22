import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_
import pytest
import statsmodels.stats.weightstats as smws
from statsmodels.tools.testing import Holder
def test_pval(self):
    assert_almost_equal(self.res1.pvalue, self.res2.p_value, decimal=13)