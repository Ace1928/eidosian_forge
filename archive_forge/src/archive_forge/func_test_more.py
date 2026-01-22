from statsmodels.compat.python import lmap
import numpy as np
import pandas
from scipy import stats
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression import gmm
from numpy.testing import assert_allclose, assert_equal
def test_more(self):
    J_df = 1
    J_p = 0.332254330027383
    J = 0.940091427212973
    j, jpval, jdf = self.res1.jtest()
    assert_allclose(jpval, J_p, rtol=5e-05, atol=0)