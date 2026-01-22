import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from sklearn.cross_decomposition._pls import (
from sklearn.datasets import load_linnerud, make_regression
from sklearn.ensemble import VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
def test_one_component_equivalence(global_random_seed):
    X, Y = make_regression(100, 10, n_targets=5, random_state=global_random_seed)
    svd = PLSSVD(n_components=1).fit(X, Y).transform(X)
    reg = PLSRegression(n_components=1).fit(X, Y).transform(X)
    canonical = PLSCanonical(n_components=1).fit(X, Y).transform(X)
    rtol = 0.001
    assert_allclose(svd, reg, atol=reg.max() * rtol)
    assert_allclose(svd, canonical, atol=canonical.max() * rtol)