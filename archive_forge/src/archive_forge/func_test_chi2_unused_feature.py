import warnings
import numpy as np
import pytest
import scipy.stats
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection._univariate_selection import _chisquare
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
def test_chi2_unused_feature():
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter('always')
        chi, p = chi2([[1, 0], [0, 0]], [1, 0])
        for w in warned:
            if 'divide by zero' in repr(w):
                raise AssertionError('Found unexpected warning %s' % w)
    assert_array_equal(chi, [1, np.nan])
    assert_array_equal(p[1], np.nan)