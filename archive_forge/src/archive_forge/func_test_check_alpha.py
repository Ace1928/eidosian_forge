import re
import warnings
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_check_alpha():
    """The provided value for alpha must only be
    used if alpha < _ALPHA_MIN and force_alpha is True.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/10772
    """
    _ALPHA_MIN = 1e-10
    b = BernoulliNB(alpha=0, force_alpha=True)
    assert b._check_alpha() == 0
    alphas = np.array([0.0, 1.0])
    b = BernoulliNB(alpha=alphas, force_alpha=True)
    b.n_features_in_ = alphas.shape[0]
    assert_array_equal(b._check_alpha(), alphas)
    msg = 'alpha too small will result in numeric errors, setting alpha = %.1e' % _ALPHA_MIN
    b = BernoulliNB(alpha=0, force_alpha=False)
    with pytest.warns(UserWarning, match=msg):
        assert b._check_alpha() == _ALPHA_MIN
    b = BernoulliNB(alpha=0, force_alpha=False)
    with pytest.warns(UserWarning, match=msg):
        assert b._check_alpha() == _ALPHA_MIN
    b = BernoulliNB(alpha=alphas, force_alpha=False)
    b.n_features_in_ = alphas.shape[0]
    with pytest.warns(UserWarning, match=msg):
        assert_array_equal(b._check_alpha(), np.array([_ALPHA_MIN, 1.0]))