import pytest
import numpy as np
from scipy import stats
from numpy.testing import assert_allclose, assert_array_less
from statsmodels.sandbox.distributions.extras import NormExpan_gen
def test_mvsk(self):
    mvsk = stats.describe(self.rvs)[-4:]
    assert_allclose(self.dist2.mvsk, mvsk, rtol=1e-12)