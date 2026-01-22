import numpy as np
import numpy.ma as ma
import scipy.stats.mstats as ms
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
def test_mquantiles_cimj(self):
    ci_lower, ci_upper = ms.mquantiles_cimj(self.data)
    assert_(ci_lower.size == ci_upper.size == 3)