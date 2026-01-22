import itertools
import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import MinCovDet, empirical_covariance, fast_mcd
from sklearn.utils._testing import assert_array_almost_equal
def test_mcd_increasing_det_warning(global_random_seed):
    X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2], [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.4, 3.4, 1.7, 0.2], [4.6, 3.6, 1.0, 0.2], [5.0, 3.0, 1.6, 0.2], [5.2, 3.5, 1.5, 0.2]]
    mcd = MinCovDet(support_fraction=0.5, random_state=global_random_seed)
    warn_msg = 'Determinant has increased'
    with pytest.warns(RuntimeWarning, match=warn_msg):
        mcd.fit(X)