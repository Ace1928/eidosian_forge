import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.special import expit
from sklearn.datasets import make_low_rank_matrix, make_sparse_spd_matrix
from sklearn.utils import gen_batches
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._testing import (
from sklearn.utils.extmath import (
from sklearn.utils.fixes import (
def test_logistic_sigmoid():

    def naive_log_logistic(x):
        return np.log(expit(x))
    x = np.linspace(-2, 2, 50)
    warn_msg = '`log_logistic` is deprecated and will be removed'
    with pytest.warns(FutureWarning, match=warn_msg):
        assert_array_almost_equal(log_logistic(x), naive_log_logistic(x))
    extreme_x = np.array([-100.0, 100.0])
    with pytest.warns(FutureWarning, match=warn_msg):
        assert_array_almost_equal(log_logistic(extreme_x), [-100, 0])