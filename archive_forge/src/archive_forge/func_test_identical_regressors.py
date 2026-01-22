import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_identical_regressors():
    newX = X.copy()
    newX[:, 1] = newX[:, 0]
    gamma = np.zeros(n_features)
    gamma[0] = gamma[1] = 1.0
    newy = np.dot(newX, gamma)
    warning_message = 'Orthogonal matching pursuit ended prematurely due to linear dependence in the dictionary. The requested precision might not have been met.'
    with pytest.warns(RuntimeWarning, match=warning_message):
        orthogonal_mp(newX, newy, n_nonzero_coefs=2)