import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm._bounds import l1_min_c
from sklearn.svm._newrand import bounded_rand_int_wrap, set_seed_wrap
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('seed, expected', [(0, 54), (_MAX_UNSIGNED_INT, 9)])
def test_newrand_set_seed(seed, expected):
    """Test that `set_seed` produces deterministic results"""
    set_seed_wrap(seed)
    generated = bounded_rand_int_wrap(100)
    assert generated == expected