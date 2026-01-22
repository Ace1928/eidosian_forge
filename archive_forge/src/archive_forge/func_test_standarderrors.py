import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
def test_standarderrors(self):
    assert_almost_equal(self.res1.bse, self.res2.bse, self.decimal_standarderrors)