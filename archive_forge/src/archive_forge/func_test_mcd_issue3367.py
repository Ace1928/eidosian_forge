import itertools
import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import MinCovDet, empirical_covariance, fast_mcd
from sklearn.utils._testing import assert_array_almost_equal
def test_mcd_issue3367(global_random_seed):
    rand_gen = np.random.RandomState(global_random_seed)
    data_values = np.linspace(-5, 5, 10).tolist()
    data = np.array(list(itertools.product(data_values, data_values)))
    data = np.hstack((data, np.zeros((data.shape[0], 1))))
    MinCovDet(random_state=rand_gen).fit(data)