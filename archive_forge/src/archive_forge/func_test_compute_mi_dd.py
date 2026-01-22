import numpy as np
import pytest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection._mutual_info import _compute_mi
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_compute_mi_dd():
    x = np.array([0, 1, 1, 0, 0])
    y = np.array([1, 0, 0, 0, 1])
    H_x = H_y = -(3 / 5) * np.log(3 / 5) - 2 / 5 * np.log(2 / 5)
    H_xy = -1 / 5 * np.log(1 / 5) - 2 / 5 * np.log(2 / 5) - 2 / 5 * np.log(2 / 5)
    I_xy = H_x + H_y - H_xy
    assert_allclose(_compute_mi(x, y, x_discrete=True, y_discrete=True), I_xy)