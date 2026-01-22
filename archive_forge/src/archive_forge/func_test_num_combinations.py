import sys
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse import random as sparse_random
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._csr_polynomial_expansion import (
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('n_features', [1, 4, 5])
@pytest.mark.parametrize('min_degree, max_degree', [(0, 1), (0, 2), (1, 3), (0, 4), (3, 4)])
@pytest.mark.parametrize('interaction_only', [True, False])
@pytest.mark.parametrize('include_bias', [True, False])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_num_combinations(n_features, min_degree, max_degree, interaction_only, include_bias, csr_container):
    """
    Test that n_output_features_ is calculated correctly.
    """
    x = csr_container(([1], ([0], [n_features - 1])))
    est = PolynomialFeatures(degree=max_degree, interaction_only=interaction_only, include_bias=include_bias)
    est.fit(x)
    num_combos = est.n_output_features_
    combos = PolynomialFeatures._combinations(n_features=n_features, min_degree=0, max_degree=max_degree, interaction_only=interaction_only, include_bias=include_bias)
    assert num_combos == sum([1 for _ in combos])