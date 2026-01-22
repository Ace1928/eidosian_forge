import itertools
import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import MinCovDet, empirical_covariance, fast_mcd
from sklearn.utils._testing import assert_array_almost_equal
def test_mcd_issue1127():
    rnd = np.random.RandomState(0)
    X = rnd.normal(size=(3, 1))
    mcd = MinCovDet()
    mcd.fit(X)