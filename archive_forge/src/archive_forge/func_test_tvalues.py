import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
def test_tvalues(self):
    if not hasattr(self.res2, 'tvalues'):
        pytest.skip('No tvalues in benchmark')
    assert_allclose(self.res1.tvalues, self.res2.tvalues, rtol=0.003)