import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_compute_mi_cc(global_dtype):
    mean = np.zeros(2)
    sigma_1 = 1
    sigma_2 = 10
    corr = 0.5
    cov = np.array([[sigma_1 ** 2, corr * sigma_1 * sigma_2], [corr * sigma_1 * sigma_2, sigma_2 ** 2]])
    I_theory = np.log(sigma_1) + np.log(sigma_2) - 0.5 * np.log(np.linalg.det(cov))
    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000).astype(global_dtype, copy=False)
    x, y = (Z[:, 0], Z[:, 1])
    for n_neighbors in [3, 5, 7]:
        I_computed = _compute_mi(x, y, x_discrete=False, y_discrete=False, n_neighbors=n_neighbors)
        assert_allclose(I_computed, I_theory, rtol=0.1)